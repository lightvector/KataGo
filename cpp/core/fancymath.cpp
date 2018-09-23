#include <cmath>
#include <sstream>
#include <iomanip>

#include "../core/fancymath.h"
#include "../core/test.h"

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
    if(fabs(mult - 1.0) <= tolerance)
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
  if(x < 0.0 || x > 1.0 || a <= 0.0 || b <= 0.0)
    return NAN;
  double logx = log(x);
  double logy = log(1-x);
  if(x <= (a+1.0)/(a+b+2.0))
    return exp(logx*a + logy*b) / a / incompleteBetaContinuedFraction(x,a,b);
  else
    return beta(a,b) - (exp(logy*b + logx*a) / b / incompleteBetaContinuedFraction(1.0-x,b,a));
}

//https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
double FancyMath::regularizedIncompleteBeta(double x, double a, double b) {
  if(x < 0.0 || x > 1.0 || a <= 0.0 || b <= 0.0)
    return NAN;
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
  return 1.0 / sqrt(v*PI) / exp(lgamma(v/2.0) - lgamma((v+1.0)/2.0)) / pow(1.0 + x*x/v, (v+1.0)/2.0);
}

double FancyMath::tdistcdf(double x, double degreesOfFreedom) {
  double v = degreesOfFreedom;
  if(x >= 0)
    return 1.0 - regularizedIncompleteBeta(v/(x*x+v), v/2.0, 0.5) / 2.0;
  else
    return regularizedIncompleteBeta(v/(x*x+v), v/2.0, 0.5) / 2.0;
}

double FancyMath::betapdf(double x, double a, double b) {
  return exp(-logbeta(a,b) + log(x)*(a-1.0) + log(1.0-x)*(b-1.0));
}

double FancyMath::betacdf(double x, double a, double b) {
  return regularizedIncompleteBeta(x,a,b);
}

void FancyMath::runTests() {
  cout << "Running fancy math tests" << endl;
  ostringstream out;
  out << std::setprecision(17);
  out << std::fixed;

  {
    const char* name = "Continued fraction tests";

    out << "Golden ratio" << endl;
    out << ((1.0 + sqrt(5)) / 2.0) << endl;
    out << evaluateContinuedFraction([](int n) { (void)n; return 1.0; }, [](int n) { (void)n; return 1.0; }, 1e-15, 1000) << endl;

    out << "Sqrt2" << endl;
    out << sqrt(2) << endl;
    out << evaluateContinuedFraction([](int n) { (void)n; return 1.0; }, [](int n) { return n == 0 ? 1.0 : 2.0; }, 1e-15, 1000) << endl;

    out << "e" << endl;
    out << exp(1) << endl;
    out << evaluateContinuedFraction([](int n) { (void)n; return 1.0; }, [](int n) { return n == 0 ? 2.0 : n%3 == 2 ? (double)((n+1)/3*2) : 1.0; }, 1e-15, 1000) << endl;

    out << "pi" << endl;
    out << PI << endl;
    out << evaluateContinuedFraction([](int n) { return (n*2-1)*(n*2-1); }, [](int n) { return n == 0 ? 3.0 : 6.0; }, 1e-15, 10000) << endl;
    out << evaluateContinuedFraction([](int n) { return n == 1 ? 4 : ((n-1)*(n-1)*4-1); }, [](int n) { return n == 0 ? 2.0 : n == 1 ? 3.0 : 4.0; }, 1e-15, 10000) << endl;
    out << evaluateContinuedFraction([](int n) { return n == 1 ? 2 : n*(n-1); }, [](int n) { return n == 0 ? 2.0 : 1.0; }, 1e-15, 10000) << endl;


    string expected = R"%%(
Golden ratio
1.61803398874989490
1.61803398874989512
Sqrt2
1.41421356237309515
1.41421356237309559
e
2.71828182845904509
2.71828182845904553
pi
3.14159265358979312
3.14159265359005602
3.14159265859030645
3.14174973714925665
)%%";
    TestCommon::expect(name,out,expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Beta tests";

    out << "a=1 b=1 uniform" << endl;
    out << "betapdf(0.00,1,1)" << betapdf(0.00,1,1) << endl;
    out << "betapdf(0.25,1,1)" << betapdf(0.25,1,1) << endl;
    out << "betapdf(0.50,1,1)" << betapdf(0.50,1,1) << endl;
    out << "betapdf(0.75,1,1)" << betapdf(0.75,1,1) << endl;
    out << "betapdf(1.00,1,1)" << betapdf(1.00,1,1) << endl;

    out << "betacdf(0.00,1,1)" << betacdf(0.00,1,1) << endl;
    out << "betacdf(0.25,1,1)" << betacdf(0.25,1,1) << endl;
    out << "betacdf(0.50,1,1)" << betacdf(0.50,1,1) << endl;
    out << "betacdf(0.75,1,1)" << betacdf(0.75,1,1) << endl;
    out << "betacdf(1.00,1,1)" << betacdf(1.00,1,1) << endl;

    out << "a=2 b=1 triangular" << endl;
    out << "betapdf(0.00,2,1)" << betapdf(0.00,2,1) << endl;
    out << "betapdf(0.25,2,1)" << betapdf(0.25,2,1) << endl;
    out << "betapdf(0.50,2,1)" << betapdf(0.50,2,1) << endl;
    out << "betapdf(0.75,2,1)" << betapdf(0.75,2,1) << endl;
    out << "betapdf(1.00,2,1)" << betapdf(1.00,2,1) << endl;

    out << "betacdf(0.00,2,1)" << betacdf(0.00,2,1) << endl;
    out << "betacdf(0.25,2,1)" << betacdf(0.25,2,1) << endl;
    out << "betacdf(0.50,2,1)" << betacdf(0.50,2,1) << endl;
    out << "betacdf(0.75,2,1)" << betacdf(0.75,2,1) << endl;
    out << "betacdf(1.00,2,1)" << betacdf(1.00,2,1) << endl;

    out << "a=3 b=1 quadratic" << endl;
    out << "betapdf(0.00,3,1)" << betapdf(0.00,3,1) << endl;
    out << "betapdf(0.25,3,1)" << betapdf(0.25,3,1) << endl;
    out << "betapdf(0.50,3,1)" << betapdf(0.50,3,1) << endl;
    out << "betapdf(0.75,3,1)" << betapdf(0.75,3,1) << endl;
    out << "betapdf(1.00,3,1)" << betapdf(1.00,3,1) << endl;

    out << "betacdf(0.00,3,1)" << betacdf(0.00,3,1) << endl;
    out << "betacdf(0.25,3,1)" << betacdf(0.25,3,1) << endl;
    out << "betacdf(0.50,3,1)" << betacdf(0.50,3,1) << endl;
    out << "betacdf(0.75,3,1)" << betacdf(0.75,3,1) << endl;
    out << "betacdf(1.00,3,1)" << betacdf(1.00,3,1) << endl;

    out << "a=0.5 b=0.5 arcsin" << endl;
    out << "betapdf(0.00,0.5,0.5)" << betapdf(0.00,0.5,0.5) << endl;
    out << "betapdf(0.25,0.5,0.5)" << betapdf(0.25,0.5,0.5) << " " << (1/PI / sqrt(0.25*(1.0-0.25))) << endl;
    out << "betapdf(0.50,0.5,0.5)" << betapdf(0.50,0.5,0.5) << " " << (1/PI / sqrt(0.50*(1.0-0.50))) << endl;
    out << "betapdf(0.75,0.5,0.5)" << betapdf(0.75,0.5,0.5) << " " << (1/PI / sqrt(0.75*(1.0-0.75))) << endl;
    out << "betapdf(1.00,0.5,0.5)" << betapdf(1.00,0.5,0.5) << endl;

    out << "betacdf(0.00,0.5,0.5)" << betacdf(0.00,0.5,0.5) << endl;
    out << "betacdf(0.25,0.5,0.5)" << betacdf(0.25,0.5,0.5) << " " << (2/PI * asin(sqrt(0.25)))  << endl;
    out << "betacdf(0.50,0.5,0.5)" << betacdf(0.50,0.5,0.5) << " " << (2/PI * asin(sqrt(0.50)))  << endl;
    out << "betacdf(0.75,0.5,0.5)" << betacdf(0.75,0.5,0.5) << " " << (2/PI * asin(sqrt(0.75)))  << endl;
    out << "betacdf(1.00,0.5,0.5)" << betacdf(1.00,0.5,0.5) << endl;

    out << "extreme values" << endl;
    out << "betapdf(0.00,.5e5,.5e1)" << betapdf(0.00,.5e5,.5e1) << endl;
    out << "betapdf(0.25,.5e5,.5e1)" << betapdf(0.25,.5e5,.5e1) << endl;
    out << "betapdf(0.50,.5e5,.5e1)" << betapdf(0.50,.5e5,.5e1) << endl;
    out << "betapdf(0.75,.5e5,.5e1)" << betapdf(0.75,.5e5,.5e1) << endl;
    out << "betapdf(1-1e-4,.5e5,.5e1)" << betapdf(1-1e-4,.5e5,.5e1) << endl;
    out << "betapdf(1.00,.5e5,.5e1)" << betapdf(1.00,.5e5,.5e1) << endl;

    out << "betacdf(0.00,.5e5,.5e1)" << betacdf(0.00,.5e5,.5e1) << endl;
    out << "betacdf(0.25,.5e5,.5e1)" << betacdf(0.25,.5e5,.5e1) << endl;
    out << "betacdf(0.50,.5e5,.5e1)" << betacdf(0.50,.5e5,.5e1) << endl;
    out << "betacdf(0.75,.5e5,.5e1)" << betacdf(0.75,.5e5,.5e1) << endl;
    out << "betacdf(1-1e-4,.5e5,.5e1)" << betacdf(1-1e-4,.5e5,.5e1) << endl;
    out << "betacdf(1.00,.5e5,.5e1)" << betacdf(1.00,.5e5,.5e1) << endl;

    out << "betapdf(0.00,.5e10,.5e2)" << betapdf(0.00,.5e10,.5e2) << endl;
    out << "betapdf(0.25,.5e10,.5e2)" << betapdf(0.25,.5e10,.5e2) << endl;
    out << "betapdf(0.50,.5e10,.5e2)" << betapdf(0.50,.5e10,.5e2) << endl;
    out << "betapdf(0.75,.5e10,.5e2)" << betapdf(0.75,.5e10,.5e2) << endl;
    out << "betapdf(1-1e-8,.5e10,.5e2)" << betapdf(1-1e-8,.5e10,.5e2) << endl;
    out << "betapdf(1.00,.5e10,.5e2)" << betapdf(1.00,.5e10,.5e2) << endl;

    out << "betacdf(0.00,.5e10,.5e2)" << betacdf(0.00,.5e10,.5e2) << endl;
    out << "betacdf(0.25,.5e10,.5e2)" << betacdf(0.25,.5e10,.5e2) << endl;
    out << "betacdf(0.50,.5e10,.5e2)" << betacdf(0.50,.5e10,.5e2) << endl;
    out << "betacdf(0.75,.5e10,.5e2)" << betacdf(0.75,.5e10,.5e2) << endl;
    out << "betacdf(1-1e-8,.5e10,.5e2)" << betacdf(1-1e-8,.5e10,.5e2) << endl;
    out << "betacdf(1.00,.5e10,.5e2)" << betacdf(1.00,.5e10,.5e2) << endl;

    //These probably aren't very accurate, we're hitting numerical instability
    out << "betapdf(0.00,.5e15,.5e3)" << betapdf(0.00,.5e15,.5e3) << endl;
    out << "betapdf(0.25,.5e15,.5e3)" << betapdf(0.25,.5e15,.5e3) << endl;
    out << "betapdf(0.50,.5e15,.5e3)" << betapdf(0.50,.5e15,.5e3) << endl;
    out << "betapdf(0.75,.5e15,.5e3)" << betapdf(0.75,.5e15,.5e3) << endl;
    out << "betapdf(1-1e-12,.5e15,.5e3)" << betapdf(1-1e-12,.5e15,.5e3) << endl;
    out << "betapdf(1.00,.5e15,.5e3)" << betapdf(1.00,.5e15,.5e3) << endl;

    out << "betacdf(0.00,.5e15,.5e3)" << betacdf(0.00,.5e15,.5e3) << endl;
    out << "betacdf(0.25,.5e15,.5e3)" << betacdf(0.25,.5e15,.5e3) << endl;
    out << "betacdf(0.50,.5e15,.5e3)" << betacdf(0.50,.5e15,.5e3) << endl;
    out << "betacdf(0.75,.5e15,.5e3)" << betacdf(0.75,.5e15,.5e3) << endl;
    out << "betacdf(1-1e-12,.5e15,.5e3)" << betacdf(1-1e-12,.5e15,.5e3) << endl;
    out << "betacdf(1.00,.5e15,.5e3)" << betacdf(1.00,.5e15,.5e3) << endl;


    string expected = R"%%(
a=1 b=1 uniform
betapdf(0.00,1,1)-nan
betapdf(0.25,1,1)1.00000000000000000
betapdf(0.50,1,1)1.00000000000000000
betapdf(0.75,1,1)1.00000000000000000
betapdf(1.00,1,1)-nan
betacdf(0.00,1,1)0.00000000000000000
betacdf(0.25,1,1)0.25000000000000000
betacdf(0.50,1,1)0.50000000000000000
betacdf(0.75,1,1)0.75000000000000000
betacdf(1.00,1,1)1.00000000000000000
a=2 b=1 triangular
betapdf(0.00,2,1)0.00000000000000000
betapdf(0.25,2,1)0.50000000000000000
betapdf(0.50,2,1)1.00000000000000000
betapdf(0.75,2,1)1.50000000000000000
betapdf(1.00,2,1)-nan
betacdf(0.00,2,1)0.00000000000000000
betacdf(0.25,2,1)0.06250000000000001
betacdf(0.50,2,1)0.25000000000000006
betacdf(0.75,2,1)0.56250000000000000
betacdf(1.00,2,1)1.00000000000000000
a=3 b=1 quadratic
betapdf(0.00,3,1)0.00000000000000000
betapdf(0.25,3,1)0.18750000000000000
betapdf(0.50,3,1)0.74999999999999989
betapdf(0.75,3,1)1.68749999999999978
betapdf(1.00,3,1)-nan
betacdf(0.00,3,1)0.00000000000000000
betacdf(0.25,3,1)0.01562500000000001
betacdf(0.50,3,1)0.12500000000000000
betacdf(0.75,3,1)0.42187500000000000
betacdf(1.00,3,1)1.00000000000000000
a=0.5 b=0.5 arcsin
betapdf(0.00,0.5,0.5)inf
betapdf(0.25,0.5,0.5)0.73510519389572271 0.73510519389572282
betapdf(0.50,0.5,0.5)0.63661977236758138 0.63661977236758138
betapdf(0.75,0.5,0.5)0.73510519389572271 0.73510519389572282
betapdf(1.00,0.5,0.5)inf
betacdf(0.00,0.5,0.5)0.00000000000000000
betacdf(0.25,0.5,0.5)0.33333333333333331 0.33333333333333337
betacdf(0.50,0.5,0.5)0.50000000000000044 0.50000000000000011
betacdf(0.75,0.5,0.5)0.66666666666666674 0.66666666666666663
betacdf(1.00,0.5,0.5)1.00000000000000000
extreme values
betapdf(0.00,.5e5,.5e1)0.00000000000000000
betapdf(0.25,.5e5,.5e1)0.00000000000000000
betapdf(0.50,.5e5,.5e1)0.00000000000000000
betapdf(0.75,.5e5,.5e1)0.00000000000000000
betapdf(1-1e-4,.5e5,.5e1)8773.80701229644182604
betapdf(1.00,.5e5,.5e1)0.00000000000000000
betacdf(0.00,.5e5,.5e1)0.00000000000000000
betacdf(0.25,.5e5,.5e1)0.00000000000000000
betacdf(0.50,.5e5,.5e1)0.00000000000000000
betacdf(0.75,.5e5,.5e1)0.00000000000000000
betacdf(1-1e-4,.5e5,.5e1)0.44041432429729233
betacdf(1.00,.5e5,.5e1)1.00000000000000000
betapdf(0.00,.5e10,.5e2)0.00000000000000000
betapdf(0.25,.5e10,.5e2)0.00000000000000000
betapdf(0.50,.5e10,.5e2)0.00000000000000000
betapdf(0.75,.5e10,.5e2)0.00000000000000000
betapdf(1-1e-8,.5e10,.5e2)281620447.51994127035140991
betapdf(1.00,.5e10,.5e2)0.00000000000000000
betacdf(0.00,.5e10,.5e2)0.00000000000000000
betacdf(0.25,.5e10,.5e2)0.00000000000000000
betacdf(0.50,.5e10,.5e2)0.00000000000000000
betacdf(0.75,.5e10,.5e2)0.00000000000000000
betacdf(1-1e-8,.5e10,.5e2)0.48120008730261921
betacdf(1.00,.5e10,.5e2)1.00000000000000000
betapdf(0.00,.5e15,.5e3)0.00000000000000000
betapdf(0.25,.5e15,.5e3)0.00000000000000000
betapdf(0.50,.5e15,.5e3)0.00000000000000000
betapdf(0.75,.5e15,.5e3)0.00000000000000000
betapdf(1-1e-12,.5e15,.5e3)12054813431812.26562500000000000
betapdf(1.00,.5e15,.5e3)0.00000000000000000
betacdf(0.00,.5e15,.5e3)0.00000000000000000
betacdf(0.25,.5e15,.5e3)0.00000000000000000
betacdf(0.50,.5e15,.5e3)0.00000000000000000
betacdf(0.75,.5e15,.5e3)0.00000000000000000
betacdf(1-1e-12,.5e15,.5e3)0.31645988794179647
betacdf(1.00,.5e15,.5e3)1.00000000000000000

)%%";
    TestCommon::expect(name,out,expected);
    out.str("");
    out.clear();
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
0.00860296989685921 0.05256845671125344
0.00950462484872471 0.05528125937750891
0.01055404131909120 0.05828598340184223
0.01178489026967015 0.06163179450865471
0.01324084385123921 0.06537938295567026
0.01497928876159015 0.06960448727306394
0.01707671063217761 0.07440276529861724
0.01963663702552687 0.07989663661717004
0.02280156777820850 0.08624506109307921
0.02677122676062158 0.09365777093187939
0.03183098861837906 0.10241638234956671
0.03839684996185653 0.11290631572034970
0.04708726126979150 0.12566591637800242
0.05883731722436056 0.14146302812150896
0.07507308636410158 0.16141446721709513
0.09794150344116635 0.18716704181099880
0.13045487138679943 0.22114206162369540
0.17586181557115499 0.26673770835657395
0.23405138689984592 0.32797913037736914
0.29202741851723918 0.40722642092225775
0.31830988618379064 0.50000000000000000
0.29202741851723918 0.59277357907774220
0.23405138689984611 0.67202086962263063
0.17586181557115513 0.73326229164342582
0.13045487138679951 0.77885793837630435
0.09794150344116635 0.81283295818900125
0.07507308636410158 0.83858553278290482
0.05883731722436058 0.85853697187849098
0.04708726126979150 0.87433408362199760
0.03839684996185655 0.88709368427965030
0.03183098861837906 0.89758361765043326
0.02677122676062160 0.90634222906812067
0.02280156777820850 0.91375493890692083
0.01963663702552687 0.92010336338282994
0.01707671063217761 0.92559723470138278
0.01497928876159015 0.93039551272693610
0.01324084385123922 0.93462061704432975
0.01178489026967015 0.93836820549134525
0.01055404131909120 0.94171401659815779
0.00950462484872472 0.94471874062249106
0.00860296989685921 0.94743154328874657
2 degrees of freedom
0.00426898476659901 0.01333573660771238
0.00493696681992815 0.01471344098493285
0.00574915247031465 0.01631230436748744
0.00674575147137829 0.01818132828107010
0.00798083832844880 0.02038353981354107
0.00952807083151784 0.02300095399713799
0.01148914670077709 0.02614163347314959
0.01400646997100218 0.02994987100815219
0.01728234258004743 0.03462107900448277
0.02160830115420297 0.04042384690183400
0.02741012223434215 0.04773298313335459
0.03531640015741586 0.05707936742576900
0.04626019063258621 0.06922510482935712
0.06161876018200969 0.08527493459498382
0.08336870769666396 0.10683317450253303
0.11413441178180375 0.13619656244550060
0.15673368198174187 0.17650168038968472
0.21229536878003327 0.23155250617764783
0.27582396394242337 0.30471663352876399
0.33096385830912672 0.39624283042008890
0.35355339059327379 0.50000000000000000
0.33096385830912672 0.60375716957991110
0.27582396394242353 0.69528336647123568
0.21229536878003341 0.76844749382235200
0.15673368198174198 0.82349831961031505
0.11413441178180375 0.86380343755449940
0.08336870769666396 0.89316682549746695
0.06161876018200972 0.91472506540501619
0.04626019063258621 0.93077489517064294
0.03531640015741588 0.94292063257423098
0.02741012223434215 0.95226701686664539
0.02160830115420299 0.95957615309816591
0.01728234258004744 0.96537892099551725
0.01400646997100218 0.97005012899184784
0.01148914670077710 0.97385836652685043
0.00952807083151784 0.97699904600286203
0.00798083832844880 0.97961646018645887
0.00674575147137829 0.98181867171892989
0.00574915247031465 0.98368769563251257
0.00493696681992815 0.98528655901506712
0.00426898476659901 0.98666426339228763
3.4 degrees of freedom
0.00169262222379825 0.00321399392939766
0.00207830900076044 0.00377725094988260
0.00257481514584431 0.00447202548305560
0.00322079195594726 0.00533703514441874
0.00407076971626488 0.00642482440906376
0.00520263319540047 0.00780757208596374
0.00672900899327169 0.00958568514151929
0.00881474404012610 0.01190065517200583
0.01170384248524306 0.01495447437735408
0.01576089310212751 0.01903915611655849
0.02153396582287807 0.02458170717413573
0.02984700616806547 0.03221220302597344
0.04192583151491177 0.04286465594571522
0.05954145624424657 0.05791919560897546
0.08509067174369259 0.07938122601279234
0.12137944338374969 0.11004896709749495
0.17060506917322513 0.15351220050398257
0.23186282125325253 0.21363872028504532
0.29732982000897346 0.29308385420969163
0.35029649146102743 0.39080095737598308
0.37102067337737060 0.50000000000000000
0.35029649146102743 0.60919904262401692
0.29732982000897373 0.70691614579030815
0.23186282125325272 0.78636127971495451
0.17060506917322527 0.84648779949601705
0.12137944338374969 0.88995103290250510
0.08509067174369259 0.92061877398720771
0.05954145624424657 0.94208080439102448
0.04192583151491177 0.95713534405428480
0.02984700616806549 0.96778779697402650
0.02153396582287807 0.97541829282586423
0.01576089310212752 0.98096084388344151
0.01170384248524307 0.98504552562264591
0.00881474404012609 0.98809934482799422
0.00672900899327170 0.99041431485848075
0.00520263319540047 0.99219242791403628
0.00407076971626488 0.99357517559093622
0.00322079195594726 0.99466296485558126
0.00257481514584431 0.99552797451694441
0.00207830900076044 0.99622274905011743
0.00169262222379825 0.99678600607060230
12.3 degrees of freedom
0.00004382411579443 0.00002803584035964
0.00007237815640093 0.00004509236024620
0.00012098366924326 0.00007344704754957
0.00020461420059234 0.00012114765375801
0.00034993634913418 0.00020231743388425
0.00060464084166131 0.00034192688622859
0.00105411292503461 0.00058436339143855
0.00185075595666021 0.00100873359114971
0.00326422306787927 0.00175585160204256
0.00576383584424745 0.00307481592544347
0.01014465352718061 0.00540062703499853
0.01769834197856321 0.00947660311360492
0.03039404983215383 0.01653119637282641
0.05095281588905719 0.02849747820452129
0.08256810849968406 0.04820985261857516
0.12791535378603000 0.07941942338275537
0.18721895176751177 0.12637139430023470
0.25580185067428807 0.19270110580754501
0.32268400543384146 0.27969913116919981
0.37242377207821037 0.38459344284542646
0.39092423129728382 0.50000000000000000
0.37242377207821037 0.61540655715457349
0.32268400543384146 0.72030086883079980
0.25580185067428846 0.80729889419245482
0.18721895176751199 0.87362860569976508
0.12791535378603000 0.92058057661724457
0.08256810849968406 0.95179014738142487
0.05095281588905719 0.97150252179547869
0.03039404983215383 0.98346880362717359
0.01769834197856324 0.99052339688639501
0.01014465352718061 0.99459937296500145
0.00576383584424746 0.99692518407455655
0.00326422306787927 0.99824414839795739
0.00185075595666021 0.99899126640885028
0.00105411292503461 0.99941563660856148
0.00060464084166131 0.99965807311377142
0.00034993634913418 0.99979768256611579
0.00020461420059234 0.99987885234624196
0.00012098366924326 0.99992655295245048
0.00007237815640093 0.99995490763975381
0.00004382411579443 0.99997196415964040

)%%";
    TestCommon::expect(name,out,expected);
    out.str("");
    out.clear();

  }
}
