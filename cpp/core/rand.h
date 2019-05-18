/*
 * rand.h
 * Author: David Wu
 *
 * Basic class for random number generation.
 * Individual rand objects are not threadsafe but the seeding logic should ensure that different
 * instances created simultaneously in different threads or on different machines should all be distinct.
 * Note: Signed integer functions might not be portable to other architectures.
 *
 * Combines:
 * PCG32 (period 2^64)
 * XorShift1024Mult (period 2^1024-1)
 */

#ifndef CORE_RAND_H_
#define CORE_RAND_H_

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdint.h>

#include "../core/rand_helpers.h"

class Rand
{
 private:
  XorShift1024Mult xorm;
  PCG32 pcg32;

  bool hasGaussian;
  double storedGaussian;

  std::string initSeed;
  uint64_t numCalls;

 public:

  //Initializes according to system time and some other unique junk, tries
  //to make sure multiple invocations will be different
  Rand();
  //Intializes according to the provided seed
  Rand(const char* seed);
  Rand(const std::string& seed);
  Rand(uint64_t seed);

  //Reinitialize according to system time and some other unique junk
  void init();
  //Reinitialize according to the provided seed
  void init(const char* seed);
  void init(const std::string& seed);
  void init(uint64_t seed);

  ~Rand();

  Rand(const Rand&) = delete;
  Rand& operator=(const Rand&) = delete;

  public:

  //MISC-------------------------------------------------
  std::string getSeed() const;
  uint64_t getNumCalls() const;

  //UNSIGNED INTEGER-------------------------------------

  //Returns a random integer in [0,2^32)
  uint32_t nextUInt();
  //Returns a random integer in [0,n)
  uint32_t nextUInt(uint32_t n);
  //Returns a random integer according to the given frequency distribution
  uint32_t nextUInt(const int* freq, size_t n);
  //Returns a random integer according to the given unnormalized probability distribution
  uint32_t nextUInt(const double* relProbs, size_t n);

  //SIGNED INTEGER---------------------------------------

  //Returns a random integer in [-2^31,2^31)
  int32_t nextInt();
  //Returns a random integer in [a,b]
  int32_t nextInt(int32_t a, int32_t b);

  //64-BIT INTEGER--------------------------------------

  //Returns a random integer in [0,2^64)
  uint64_t nextUInt64();
  //Returns a random integer in [0,n)
  uint64_t nextUInt64(uint64_t n);

  //BOOL------------------------------------------------
  //Returns true with probability prob
  bool nextBool(double prob);

  //DOUBLE----------------------------------------------

  //Returns a random double in [0,1)
  double nextDouble();
  //Returns a random double in [0,n). Note: Rarely, it may be possible for n to occur.
  double nextDouble(double n);
  //Returns a random double in [a,b). Note: Rarely, it may be possible for b to occur.
  double nextDouble(double a, double b);

  //Returns a normally distributed double with mean 0 stdev 1
  double nextGaussian();
  //Returns an exponentially distributed double with mean 1.
  double nextExponential();
  //Returns a logistically distributed double with mean 0 and scale 1 (cdf = 1/(1+exp(-x)))
  double nextLogistic();
  //Returns a gamma distributed double with shape a and scale 1
  double nextGamma(double a);

  //TESTING----------------------------------------------
  static void runTests();
};

inline std::string Rand::getSeed() const
{
  return initSeed;
}

inline uint64_t Rand::getNumCalls() const
{
  return numCalls;
}

inline uint32_t Rand::nextUInt()
{
  return pcg32.nextUInt() + xorm.nextUInt();
}

inline uint32_t Rand::nextUInt(uint32_t n)
{
  assert(n > 0);
  uint32_t bits, val;
  do {
    bits = nextUInt();
    val = bits % n;
  } while((uint32_t)(bits - val + (n-1)) < (uint32_t)(bits - val)); //If adding (n-1) overflows, no good.
  return val;
}

inline int32_t Rand::nextInt()
{
  return (int32_t)nextUInt();
}

inline int32_t Rand::nextInt(int32_t a, int32_t b)
{
  assert(b >= a);
  uint32_t max = (uint32_t)b-(uint32_t)a+(uint32_t)1;
  if(max == 0)
    return (int32_t)(nextUInt());
  else
    return (int32_t)(nextUInt(max)+(uint32_t)a);
}

inline uint64_t Rand::nextUInt64()
{
  return ((uint64_t)nextUInt()) | ((uint64_t)nextUInt() << 32);
}

inline uint64_t Rand::nextUInt64(uint64_t n)
{
  assert(n > 0);
  uint64_t bits, val;
  do {
    bits = nextUInt64();
    val = bits % n;
  } while((uint64_t)(bits - val + (n-1)) < (uint64_t)(bits - val)); //If adding (n-1) overflows, no good.
  return val;
}

inline uint32_t Rand::nextUInt(const int* freq, size_t n)
{
  assert(n > 0);
  assert(n < 0xFFFFFFFF);
  int64_t sum = 0;
  for(uint32_t i = 0; i<n; i++)
  {
    assert(freq[i] >= 0);
    sum += freq[i];
  }

  int64_t r = (int64_t)nextUInt64(sum);
  assert(r >= 0);
  for(uint32_t i = 0; i<n; i++)
  {
    r -= freq[i];
    if(r < 0)
    {return i;}
  }
  //Should not get to here
  assert(false);
  return 0;
}

inline uint32_t Rand::nextUInt(const double* relProbs, size_t n)
{
  assert(n > 0);
  assert(n < 0xFFFFFFFF);
  double_t sum = 0;
  for(uint32_t i = 0; i<n; i++)
  {
    assert(relProbs[i] >= 0);
    sum += relProbs[i];
  }

  double d = nextDouble(sum);
  sum = 0.0;
  for(uint32_t i = 0; i<n; i++)
  {
    sum += relProbs[i];
    if(sum > d)
      return i;
  }
  return (uint32_t)(n-1);
}


inline double Rand::nextDouble()
{
  double x;
  do
  {
    uint64_t bits = nextUInt64() & ((1ULL << 53)-1ULL);
    x = (double)bits / (double)(1ULL << 53);
  }
  //Catch loss of precision of long --> double conversion
  while (!(x>=0.0 && x<1.0));

  return x;
}

inline double Rand::nextDouble(double n)
{
  assert(n >= 0);
  return nextDouble()*n;
}

inline double Rand::nextDouble(double a, double b)
{
  assert(b >= a);
  return a+nextDouble(b-a);
}

inline bool Rand::nextBool(double prob)
{
  return nextDouble() < prob;
}

inline double Rand::nextGaussian()
{
  if(hasGaussian)
  {
    hasGaussian = false;
    return storedGaussian;
  }
  else
  {
    double v1, v2, s;
    do
    {
      v1 = nextDouble() * 2.0 - 1.0;
      v2 = nextDouble() * 2.0 - 1.0;
      s = v1 * v1 + v2 * v2;
    } while (s >= 1 || s == 0);

    double multiplier = sqrt(-2 * log(s)/s);
    storedGaussian = v2 * multiplier;
    hasGaussian = true;
    return v1 * multiplier;
  }
}

inline double Rand::nextExponential()
{
  double r = 0.0;
  while(r <= 1e-16)
    r = nextDouble();
  return -log(r);
}

inline double Rand::nextLogistic()
{
  double x = nextDouble();
  return log(x / (1.0 - x));
}

#endif  // CORE_RAND_H_
