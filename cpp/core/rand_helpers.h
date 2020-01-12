#ifndef CORE_RAND_HELPERS_H_
#define CORE_RAND_HELPERS_H_

#include <stdint.h>

//-------------------------------------------------------------------------------------

//xorshift1024* from http://xoroshiro.di.unimi.it/
//Not all values should be zero
//Period = 2^1024 - 1
//Not threadsafe
class XorShift1024Mult
{
 public:
  static const int XORMULT_LEN = 16;
  static const int XORMULT_MASK = XORMULT_LEN-1;

  XorShift1024Mult(const uint64_t* init_a);
  void init(const uint64_t* init_a);
  uint32_t nextUInt();

  static void test();

 private:
  uint64_t a[XORMULT_LEN];
  uint64_t a_idx;
};

inline uint32_t XorShift1024Mult::nextUInt()
{
  uint64_t a0 = a[a_idx];
  uint64_t a1 = a[a_idx = (a_idx + 1) & XORMULT_MASK];
  a1 ^= a1 << 31; // a
  a1 ^= a1 >> 11; // b
  a0 ^= a0 >> 30; // c
  a[a_idx] = a0 ^ a1;
  uint64_t a_result = a[a_idx] * 1181783497276652981LL;

  return (uint32_t)(a_result >> 32);
}

//-------------------------------------------------------------------------------------

//PCG Generator from http://www.pcg-random.org/
//Period = 2^64
//Not threadsafe
class PCG32
{
 public:
  PCG32(uint64_t state);
  void init(uint64_t state);
  uint32_t nextUInt();

  static void test();

 private:
  uint64_t s;
};

inline uint32_t PCG32::nextUInt()
{
  s = s * 6364136223846793005ULL + 1442695040888963407ULL;
  uint32_t x = (uint32_t)(((s >> 18) ^ s) >> 27);
  int rot = (int)(s >> 59);
  return rot == 0 ? x : ((x >> rot) | (x << (32-rot)));
}

#endif  // CORE_RAND_HELPERS_H_
