#include "../core/fancymath.h"

#include <cmath>
#include <iomanip>
#include <sstream>

#include "../core/test.h"

using namespace std;

//https://en.wikipedia.org/wiki/Beta_function
double FancyMath::beta(double a, double b) {
  return exp(lgamma(a) + lgamma(b) - lgamma(a+b));
}
double FancyMath::logbeta(double a, double b) {
  return lgamma(a) + lgamma(b) - lgamma(a+b);
}

//Modified Lentz Algorithm
static double evaluateContinuedFractionHelper(const function<double(int)>& numer, const function<double(int)>& denom, double tolerance, int maxTerms) {
  double tiny = 1e-300;
  double ret = denom(0);
  if(ret == 0.0)
    ret = tiny;
  double c = ret;
  double d = 0.0;

  int n;
  for(n = 1; n < maxTerms; n++) {
    double nextnumer = numer(n);
    double nextdenom = denom(n);
    d = nextdenom + nextnumer*d;
    if(d == 0.0)
      d = tiny;
    c = nextdenom + nextnumer/c;
    if(c == 0)
      c = tiny;
    d = 1.0/d;
    double mult = c*d;
    ret = ret * mult;
    if(std::fabs(mult - 1.0) <= tolerance)
      break;
  }
  return ret;
}

double FancyMath::evaluateContinuedFraction(const function<double(int)>& numer, const function<double(int)>& denom, double tolerance, int maxTerms) {
  return evaluateContinuedFractionHelper(numer,denom,tolerance,maxTerms);
}

//Textbook continued fraction term for incomplete beta function
static double incompleteBetaContinuedFraction(double x, double a, double b) {
  auto numer = [x,a,b](int n) {
    if(n % 2 == 0) {
      double m = n / 2;
      return m * (b-m) * x / (a + 2.0*m - 1.0) / (a + 2.0*m);
    }
    else {
      double m = (n-1) / 2;
      return -(a+m) * (a+b+m) * x / (a + 2.0*m) / (a + 2.0*m + 1.0);
    }
  };
  auto denom = [](int n) { (void)n; return 1.0; };
  return evaluateContinuedFractionHelper(numer, denom, 1e-15, 100000);
}

//https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
double FancyMath::incompleteBeta(double x, double a, double b) {
  if(!(x >= 0.0 && x <= 1.0 && a > 0.0 && b > 0.0))
    return NAN;
  if(x <= 0.0)
    return 0.0;
  if(x >= 1.0)
    return beta(a,b);
  double logx = log(x);
  double logy = log(1-x);
  if(x <= (a+1.0)/(a+b+2.0))
    return exp(logx*a + logy*b) / a / incompleteBetaContinuedFraction(x,a,b);
  else
    return beta(a,b) - (exp(logy*b + logx*a) / b / incompleteBetaContinuedFraction(1.0-x,b,a));
}

//https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
double FancyMath::regularizedIncompleteBeta(double x, double a, double b) {
  if(!(x >= 0.0 && x <= 1.0 && a > 0.0 && b > 0.0))
    return NAN;
  if(x <= 0.0)
    return 0.0;
  if(x >= 1.0)
    return 1.0;
  double logx = log(x);
  double logy = log(1-x);
  if(x <= (a+1.0)/(a+b+2.0))
    return exp(logx*a + logy*b - logbeta(a,b)) / a / incompleteBetaContinuedFraction(x,a,b);
  else
    return 1.0 - (exp(logy*b + logx*a - logbeta(a,b)) / b / incompleteBetaContinuedFraction(1.0-x,b,a));
}

static const double PI = 3.1415926535897932384626433832795;

double FancyMath::tdistpdf(double x, double degreesOfFreedom) {
  double v = degreesOfFreedom;
  if(!(v > 0))
    return NAN;
  return 1.0 / sqrt(v*PI) / exp(lgamma(v/2.0) - lgamma((v+1.0)/2.0)) / pow(1.0 + x*x/v, (v+1.0)/2.0);
}

double FancyMath::tdistcdf(double x, double degreesOfFreedom) {
  double v = degreesOfFreedom;
  if(!(v > 0))
    return NAN;
  if(x >= 0)
    return 1.0 - regularizedIncompleteBeta(v/(x*x+v), v/2.0, 0.5) / 2.0;
  else
    return regularizedIncompleteBeta(v/(x*x+v), v/2.0, 0.5) / 2.0;
}

double FancyMath::betapdf(double x, double a, double b) {
  if(!(x >= 0.0 && x <= 1.0 && a > 0.0 && b > 0.0))
    return NAN;
  if(x == 0.0)
    return a < 1.0 ? INFINITY : a > 1.0 ? 0.0 : 1.0/beta(a,b);
  if(x == 1.0)
    return b < 1.0 ? INFINITY : b > 1.0 ? 0.0 : 1.0/beta(a,b);
  return exp(-logbeta(a,b) + log(x)*(a-1.0) + log(1.0-x)*(b-1.0));
}

double FancyMath::betacdf(double x, double a, double b) {
  return regularizedIncompleteBeta(x,a,b);
}

double FancyMath::normToTApprox(double z, double degreesOfFreedom) {
  double n = degreesOfFreedom;
  return sqrt(n * (expm1(z * z * (n-1.5) / ((n-1) * (n-1)))));
}

double FancyMath::binaryCrossEntropy(double predProb, double targetProb, double epsilon) {
  double reverseProb = 1.0 - predProb;
  predProb = epsilon * (1.0 - predProb) + (1.0 - epsilon) * predProb;
  reverseProb = epsilon * (1.0 - reverseProb) + (1.0 - epsilon) * reverseProb;

  // Just in case there is float weirdness
  if(predProb < epsilon) predProb = epsilon;
  if(predProb > 1.0 - epsilon) predProb = 1.0 - epsilon;
  if(reverseProb < epsilon) reverseProb = epsilon;
  if(reverseProb > 1.0 - epsilon) reverseProb = 1.0 - epsilon;

  return targetProb * (-log(predProb)) + (1.0-targetProb) * (-log(reverseProb));
}


#define APPROX_EQ(x,y,tolerance) testApproxEq((x),(y),(tolerance), #x, #y, __FILE__, __LINE__)
static void testApproxEq(double x, double y, double tolerance, const char* msgX, const char* msgY, const char *file, int line) {

  double maxDiff = tolerance * std::max(std::fabs(x),std::max(std::fabs(y),1.0));
  if(std::fabs(x-y) <= maxDiff)
    return;
  Global::fatalError(
    std::string("Failed approx equal: ") + std::string(msgX) + " " + std::string(msgY) + "\n" +
    std::string("file: ") + std::string(file) + "\n" + std::string("line: ") + Global::intToString(line) + "\n" +
    std::string("Values: ") + Global::strprintf("%.17f",x) + " " + Global::strprintf("%.17f",y)
  );
}

void FancyMath::runTests() {
  cout << "Running fancy math tests" << endl;
  ostringstream out;
  out << std::setprecision(10);
  out << std::fixed;

  {
    //const char* name = "Continued fraction tests";
    double x;
    double y;

    x = (1.0 + sqrt(5)) / 2.0;
    y = evaluateContinuedFraction([](int n) { (void)n; return 1.0; }, [](int n) { (void)n; return 1.0; }, 1e-15, 1000);
    APPROX_EQ(x,y,1e-14);

    x = sqrt(2);
    y = evaluateContinuedFraction([](int n) { (void)n; return 1.0; }, [](int n) { return n == 0 ? 1.0 : 2.0; }, 1e-15, 1000);
    APPROX_EQ(x,y,1e-14);

    x = exp(1);
    y = evaluateContinuedFraction([](int n) { (void)n; return 1.0; }, [](int n) { return n == 0 ? 2.0 : n%3 == 2 ? (double)((n+1)/3*2) : 1.0; }, 1e-15, 1000);
    APPROX_EQ(x,y,1e-14);

    x = PI;
    y = evaluateContinuedFraction([](int n) { return (n*2-1)*(n*2-1); }, [](int n) { return n == 0 ? 3.0 : 6.0; }, 1e-15, 10000);
    APPROX_EQ(x,y,1e-10);
    y = evaluateContinuedFraction([](int n) { return n == 1 ? 4 : ((n-1)*(n-1)*4-1); }, [](int n) { return n == 0 ? 2.0 : n == 1 ? 3.0 : 4.0; }, 1e-15, 10000);
    APPROX_EQ(x,y,1e-8);
    y = evaluateContinuedFraction([](int n) { return n == 1 ? 2 : n*(n-1); }, [](int n) { return n == 0 ? 2.0 : 1.0; }, 1e-15, 10000);
    APPROX_EQ(x,y,1e-3);
  }

  {
    //const char* name = "normToTApprox tests";
    APPROX_EQ(normToTApprox(2,2),  3.57464854186552161, 1e-14);
    APPROX_EQ(normToTApprox(2,4),  2.85498285635306948, 1e-14);
    APPROX_EQ(normToTApprox(2,8),  2.36638591905649687, 1e-14);
    APPROX_EQ(normToTApprox(2,16), 2.16905959247696289, 1e-14);
    APPROX_EQ(normToTApprox(2,10000), 2.0002500310444534, 1e-13);
    APPROX_EQ(normToTApprox(4,2),  77.20049205855787022, 1e-14);
    APPROX_EQ(normToTApprox(4,4),  18.34694064061386953, 1e-14);
    APPROX_EQ(normToTApprox(4,8),  7.66893227341667760, 1e-14);
    APPROX_EQ(normToTApprox(4,16), 5.37279049993877056, 1e-14);
    APPROX_EQ(normToTApprox(4,10000), 4.00170065227857751, 1e-13);
    APPROX_EQ(normToTApprox(8,2),  12566858.01484839618206024, 1e-14);
    APPROX_EQ(normToTApprox(8,4),  14501.91603376931016101, 1e-14);
    APPROX_EQ(normToTApprox(8,8),  197.25867566592546609, 1e-14);
    APPROX_EQ(normToTApprox(8,16), 31.19831990116452403, 1e-14);
    APPROX_EQ(normToTApprox(8,10000), 8.01301804270851292, 1e-13);
  }

  {
    //const char* name = "Beta tests";

    //a=1 b=1 uniform
    APPROX_EQ(betapdf(0.00,1,1), 1.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.25,1,1), 1.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.50,1,1), 1.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.75,1,1), 1.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(1.00,1,1), 1.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.00,1,1), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.25,1,1), 0.25000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.50,1,1), 0.50000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.75,1,1), 0.75000000000000000, 1e-13);
    APPROX_EQ(betacdf(1.00,1,1), 1.00000000000000000, 1e-13);
    //a=2 b=1 triangular
    APPROX_EQ(betapdf(0.00,2,1), 0.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.25,2,1), 0.50000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.50,2,1), 1.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.75,2,1), 1.50000000000000000, 1e-13);
    APPROX_EQ(betapdf(1.00,2,1), 2.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.00,2,1), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.25,2,1), 0.06250000000000001, 1e-13);
    APPROX_EQ(betacdf(0.50,2,1), 0.25000000000000006, 1e-13);
    APPROX_EQ(betacdf(0.75,2,1), 0.56250000000000000, 1e-13);
    APPROX_EQ(betacdf(1.00,2,1), 1.00000000000000000, 1e-13);
    //a=3 b=1 quadratic
    APPROX_EQ(betapdf(0.00,3,1), 0.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.25,3,1), 0.18750000000000000, 1e-13);
    APPROX_EQ(betapdf(0.50,3,1), 0.75000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.75,3,1), 1.68750000000000000, 1e-13);
    APPROX_EQ(betapdf(1.00,3,1), 3.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.00,3,1), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.25,3,1), 0.01562500000000001, 1e-13);
    APPROX_EQ(betacdf(0.50,3,1), 0.12500000000000000, 1e-13);
    APPROX_EQ(betacdf(0.75,3,1), 0.42187500000000000, 1e-13);
    APPROX_EQ(betacdf(1.00,3,1), 1.00000000000000000, 1e-13);
    //a=0.5 b=0.5 arcsin
    testAssert(betapdf(0.00,0.5,0.5) >= INFINITY);
    APPROX_EQ(betapdf(0.10,0.5,0.5), (1/PI / sqrt(0.10*(1.0-0.10))), 1e-13);
    APPROX_EQ(betapdf(0.25,0.5,0.5), (1/PI / sqrt(0.25*(1.0-0.25))), 1e-13);
    APPROX_EQ(betapdf(0.50,0.5,0.5), (1/PI / sqrt(0.50*(1.0-0.50))), 1e-13);
    APPROX_EQ(betapdf(0.75,0.5,0.5), (1/PI / sqrt(0.75*(1.0-0.75))), 1e-13);
    APPROX_EQ(betapdf(0.90,0.5,0.5), (1/PI / sqrt(0.90*(1.0-0.90))), 1e-13);
    testAssert(betapdf(1.00,0.5,0.5) >= INFINITY);
    APPROX_EQ(betacdf(0.00,0.5,0.5), 0, 1e-13);
    APPROX_EQ(betacdf(0.10,0.5,0.5), (2/PI * asin(sqrt(0.10))), 1e-13);
    APPROX_EQ(betacdf(0.25,0.5,0.5), (2/PI * asin(sqrt(0.25))), 1e-13);
    APPROX_EQ(betacdf(0.50,0.5,0.5), (2/PI * asin(sqrt(0.50))), 1e-13);
    APPROX_EQ(betacdf(0.75,0.5,0.5), (2/PI * asin(sqrt(0.75))), 1e-13);
    APPROX_EQ(betacdf(0.90,0.5,0.5), (2/PI * asin(sqrt(0.90))), 1e-13);
    APPROX_EQ(betacdf(1.00,0.5,0.5), 1, 1e-13);
    //extreme values
    APPROX_EQ(betapdf(0.00,.5e5,.5e1), 0.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.25,.5e5,.5e1), 0.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.50,.5e5,.5e1), 0.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.75,.5e5,.5e1), 0.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(1-1e-4,.5e5,.5e1), 8773.80701229644182604, 1e-9);
    APPROX_EQ(betapdf(1.00,.5e5,.5e1), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.00,.5e5,.5e1), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.25,.5e5,.5e1), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.50,.5e5,.5e1), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.75,.5e5,.5e1), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(1-1e-4,.5e5,.5e1), 0.44041432429729233, 1e-9);
    APPROX_EQ(betacdf(1.00,.5e5,.5e1), 1.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.00,.5e10,.5e2), 0.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.25,.5e10,.5e2), 0.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.50,.5e10,.5e2), 0.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.75,.5e10,.5e2), 0.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(1-1e-8,.5e10,.5e2), 281620447.51994127035140991, 1e-4);
    APPROX_EQ(betapdf(1.00,.5e10,.5e2), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.00,.5e10,.5e2), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.25,.5e10,.5e2), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.50,.5e10,.5e2), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.75,.5e10,.5e2), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(1-1e-8,.5e10,.5e2), 0.48120008730261921, 1e-4);
    APPROX_EQ(betacdf(1.00,.5e10,.5e2), 1.00000000000000000, 1e-13);
    //These probably aren't very accurate, we're hitting numerical instability
    APPROX_EQ(betapdf(0.00,.5e15,.5e3), 0.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.25,.5e15,.5e3), 0.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.50,.5e15,.5e3), 0.00000000000000000, 1e-13);
    APPROX_EQ(betapdf(0.75,.5e15,.5e3), 0.00000000000000000, 1e-13);
    // APPROX_EQ(betapdf(1-1e-12,.5e15,.5e3), 12054813431812.26562500000000000, 1e-5);
    APPROX_EQ(betapdf(1.00,.5e15,.5e3), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.00,.5e15,.5e3), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.25,.5e15,.5e3), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.50,.5e15,.5e3), 0.00000000000000000, 1e-13);
    APPROX_EQ(betacdf(0.75,.5e15,.5e3), 0.00000000000000000, 1e-13);
    // APPROX_EQ(betacdf(1-1e-12,.5e15,.5e3), 0.31645988794179647, 1e-5);
    APPROX_EQ(betacdf(1.00,.5e15,.5e3), 1.00000000000000000, 1e-13);
  }

  {
    APPROX_EQ(binaryCrossEntropy(0.5,1.0,0.001), log(2.0), 1e-13);
    APPROX_EQ(binaryCrossEntropy(0.5,0.0,0.001), log(2.0), 1e-13);
    APPROX_EQ(binaryCrossEntropy(0.5,0.7,0.001), log(2.0), 1e-13);
    APPROX_EQ(binaryCrossEntropy(1.0/exp(1.0),1.000,0.0), 1.0, 1e-13);
    APPROX_EQ(binaryCrossEntropy(1.0/exp(1.0),0.000,0.0), 1.0 - log(exp(1.0)-1.0), 1e-13);
    APPROX_EQ(binaryCrossEntropy(1.0/exp(1.0),0.000,0.5), log(2.0), 1e-13);
    APPROX_EQ(binaryCrossEntropy(0.0,1.000,0.25), 2.0 * log(2.0), 1e-13);
    APPROX_EQ(binaryCrossEntropy(1.0/6.0,1.000,0.25), log(3.0), 1e-13);
    APPROX_EQ(binaryCrossEntropy(1.0/6.0,0.000,0.25), log(3.0/2.0), 1e-13);
    APPROX_EQ(binaryCrossEntropy(1.0/6.0,0.800,0.25), 0.8 * log(3.0) + 0.2 * log(3.0/2.0), 1e-13);
  }

  {
    const char* name = "T distribution tests";
    out << "1 degrees of freedom" << endl;
    for(int i = 0; i<41; i++) {
      out << tdistpdf(-6.0+i*0.3,1.0) << " " << tdistcdf(-6.0+i*0.3,1.0) << endl;
    }
    out << "2 degrees of freedom" << endl;
    for(int i = 0; i<41; i++) {
      out << tdistpdf(-6.0+i*0.3,2.0) << " " << tdistcdf(-6.0+i*0.3,2.0) << endl;
    }
    out << "3.4 degrees of freedom" << endl;
    for(int i = 0; i<41; i++) {
      out << tdistpdf(-6.0+i*0.3,3.4) << " " << tdistcdf(-6.0+i*0.3,3.4) << endl;
    }
    out << "12.3 degrees of freedom" << endl;
    for(int i = 0; i<41; i++) {
      out << tdistpdf(-6.0+i*0.3,12.3) << " " << tdistcdf(-6.0+i*0.3,12.3) << endl;
    }

    string expected = R"%%(
1 degrees of freedom
0.0086029699 0.0525684567
0.0095046248 0.0552812594
0.0105540413 0.0582859834
0.0117848903 0.0616317945
0.0132408439 0.0653793830
0.0149792888 0.0696044873
0.0170767106 0.0744027653
0.0196366370 0.0798966366
0.0228015678 0.0862450611
0.0267712268 0.0936577709
0.0318309886 0.1024163823
0.0383968500 0.1129063157
0.0470872613 0.1256659164
0.0588373172 0.1414630281
0.0750730864 0.1614144672
0.0979415034 0.1871670418
0.1304548714 0.2211420616
0.1758618156 0.2667377084
0.2340513869 0.3279791304
0.2920274185 0.4072264209
0.3183098862 0.5000000000
0.2920274185 0.5927735791
0.2340513869 0.6720208696
0.1758618156 0.7332622916
0.1304548714 0.7788579384
0.0979415034 0.8128329582
0.0750730864 0.8385855328
0.0588373172 0.8585369719
0.0470872613 0.8743340836
0.0383968500 0.8870936843
0.0318309886 0.8975836177
0.0267712268 0.9063422291
0.0228015678 0.9137549389
0.0196366370 0.9201033634
0.0170767106 0.9255972347
0.0149792888 0.9303955127
0.0132408439 0.9346206170
0.0117848903 0.9383682055
0.0105540413 0.9417140166
0.0095046248 0.9447187406
0.0086029699 0.9474315433
2 degrees of freedom
0.0042689848 0.0133357366
0.0049369668 0.0147134410
0.0057491525 0.0163123044
0.0067457515 0.0181813283
0.0079808383 0.0203835398
0.0095280708 0.0230009540
0.0114891467 0.0261416335
0.0140064700 0.0299498710
0.0172823426 0.0346210790
0.0216083012 0.0404238469
0.0274101222 0.0477329831
0.0353164002 0.0570793674
0.0462601906 0.0692251048
0.0616187602 0.0852749346
0.0833687077 0.1068331745
0.1141344118 0.1361965624
0.1567336820 0.1765016804
0.2122953688 0.2315525062
0.2758239639 0.3047166335
0.3309638583 0.3962428304
0.3535533906 0.5000000000
0.3309638583 0.6037571696
0.2758239639 0.6952833665
0.2122953688 0.7684474938
0.1567336820 0.8234983196
0.1141344118 0.8638034376
0.0833687077 0.8931668255
0.0616187602 0.9147250654
0.0462601906 0.9307748952
0.0353164002 0.9429206326
0.0274101222 0.9522670169
0.0216083012 0.9595761531
0.0172823426 0.9653789210
0.0140064700 0.9700501290
0.0114891467 0.9738583665
0.0095280708 0.9769990460
0.0079808383 0.9796164602
0.0067457515 0.9818186717
0.0057491525 0.9836876956
0.0049369668 0.9852865590
0.0042689848 0.9866642634
3.4 degrees of freedom
0.0016926222 0.0032139939
0.0020783090 0.0037772509
0.0025748151 0.0044720255
0.0032207920 0.0053370351
0.0040707697 0.0064248244
0.0052026332 0.0078075721
0.0067290090 0.0095856851
0.0088147440 0.0119006552
0.0117038425 0.0149544744
0.0157608931 0.0190391561
0.0215339658 0.0245817072
0.0298470062 0.0322122030
0.0419258315 0.0428646559
0.0595414562 0.0579191956
0.0850906717 0.0793812260
0.1213794434 0.1100489671
0.1706050692 0.1535122005
0.2318628213 0.2136387203
0.2973298200 0.2930838542
0.3502964915 0.3908009574
0.3710206734 0.5000000000
0.3502964915 0.6091990426
0.2973298200 0.7069161458
0.2318628213 0.7863612797
0.1706050692 0.8464877995
0.1213794434 0.8899510329
0.0850906717 0.9206187740
0.0595414562 0.9420808044
0.0419258315 0.9571353441
0.0298470062 0.9677877970
0.0215339658 0.9754182928
0.0157608931 0.9809608439
0.0117038425 0.9850455256
0.0088147440 0.9880993448
0.0067290090 0.9904143149
0.0052026332 0.9921924279
0.0040707697 0.9935751756
0.0032207920 0.9946629649
0.0025748151 0.9955279745
0.0020783090 0.9962227491
0.0016926222 0.9967860061
12.3 degrees of freedom
0.0000438241 0.0000280358
0.0000723782 0.0000450924
0.0001209837 0.0000734470
0.0002046142 0.0001211477
0.0003499363 0.0002023174
0.0006046408 0.0003419269
0.0010541129 0.0005843634
0.0018507560 0.0010087336
0.0032642231 0.0017558516
0.0057638358 0.0030748159
0.0101446535 0.0054006270
0.0176983420 0.0094766031
0.0303940498 0.0165311964
0.0509528159 0.0284974782
0.0825681085 0.0482098526
0.1279153538 0.0794194234
0.1872189518 0.1263713943
0.2558018507 0.1927011058
0.3226840054 0.2796991312
0.3724237721 0.3845934428
0.3909242313 0.5000000000
0.3724237721 0.6154065572
0.3226840054 0.7203008688
0.2558018507 0.8072988942
0.1872189518 0.8736286057
0.1279153538 0.9205805766
0.0825681085 0.9517901474
0.0509528159 0.9715025218
0.0303940498 0.9834688036
0.0176983420 0.9905233969
0.0101446535 0.9945993730
0.0057638358 0.9969251841
0.0032642231 0.9982441484
0.0018507560 0.9989912664
0.0010541129 0.9994156366
0.0006046408 0.9996580731
0.0003499363 0.9997976826
0.0002046142 0.9998788523
0.0001209837 0.9999265530
0.0000723782 0.9999549076
0.0000438241 0.9999719642

)%%";
    TestCommon::expect(name,out,expected);

  }
}
