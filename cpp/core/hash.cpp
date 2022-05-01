#include "../core/hash.h"

/*
 * hash.cpp
 * Author: David Wu
 */

using namespace std;

//BITS-----------------------------------

uint32_t Hash::highBits(uint64_t x)
{
  return (uint32_t)((x >> 32) & 0xFFFFFFFFU);
}

uint32_t Hash::lowBits(uint64_t x)
{
  return (uint32_t)(x & 0xFFFFFFFFU);
}

uint64_t Hash::combine(uint32_t hi, uint32_t lo)
{
  return ((uint64_t)hi << 32) | (uint64_t)lo;
}

//A simple 64 bit linear congruential
uint64_t Hash::basicLCong(uint64_t x)
{
  return 2862933555777941757ULL*x + 3037000493ULL;
}
uint64_t Hash::basicLCong2(uint64_t x)
{
  return 6364136223846793005ULL*x + 1442695040888963407ULL;
}

//MurmurHash3 finalization - good avalanche properties
//Reversible, but maps 0 -> 0
uint64_t Hash::murmurMix(uint64_t x)
{
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

//Splitmix64 mixing step
uint64_t Hash::splitMix64(uint64_t x)
{
  x = x + 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static uint64_t rotateRight(uint64_t x, int r) {
  return (x >> r) | (x << (64 - r));
}

//algorithm from Pelle Evensen https://mostlymangling.blogspot.com/
uint64_t Hash::rrmxmx(uint64_t x)
{
  x ^= rotateRight(x,49) ^ rotateRight(x,24);
  x *= 0x9fb21c651e98df25ULL;
  x ^= x >> 28;
  x *= 0x9fb21c651e98df25ULL;
  return x ^ (x >> 28);
}
//algorithm from Pelle Evensen https://mostlymangling.blogspot.com/
uint64_t Hash::nasam(uint64_t x)
{
  x ^= rotateRight(x,25) ^ rotateRight(x,47);
  x *= 0x9e6c63d0676a9a99ULL;
  x ^= (x >> 23) ^ (x >> 51);
  x *= 0x9e6d62d06f6a9a9bULL;
  x ^= (x >> 23) ^ (x >> 51);
  return x;
}


//Robert Jenkins' 96 bit Mix Function
uint32_t Hash::jenkinsMixSingle(uint32_t a, uint32_t b, uint32_t c)
{
  a=a-b;  a=a-c;  a=a^(c >> 13);
  b=b-c;  b=b-a;  b=b^(a << 8);
  c=c-a;  c=c-b;  c=c^(b >> 13);
  a=a-b;  a=a-c;  a=a^(c >> 12);
  b=b-c;  b=b-a;  b=b^(a << 16);
  c=c-a;  c=c-b;  c=c^(b >> 5);
  a=a-b;  a=a-c;  a=a^(c >> 3);
  b=b-c;  b=b-a;  b=b^(a << 10);
  c=c-a;  c=c-b;  c=c^(b >> 15);
  return c;
}
void Hash::jenkinsMix(uint32_t& a, uint32_t& b, uint32_t& c)
{
  a=a-b;  a=a-c;  a=a^(c >> 13);
  b=b-c;  b=b-a;  b=b^(a << 8);
  c=c-a;  c=c-b;  c=c^(b >> 13);
  a=a-b;  a=a-c;  a=a^(c >> 12);
  b=b-c;  b=b-a;  b=b^(a << 16);
  c=c-a;  c=c-b;  c=c^(b >> 5);
  a=a-b;  a=a-c;  a=a^(c >> 3);
  b=b-c;  b=b-a;  b=b^(a << 10);
  c=c-a;  c=c-b;  c=c^(b >> 15);
}

uint64_t Hash::simpleHash(const char* str)
{
  uint64_t m1 = 123456789;
  uint64_t m2 = 314159265;
  uint64_t m3 = 958473711;
  while(*str != '\0')
  {
    char c = *str;
    m1 = m1 * 31 + (uint64_t)c;
    m2 = m2 * 317 + (uint64_t)c;
    m3 = m3 * 1609 + (uint64_t)c;
    str++;
  }
  uint32_t lo = jenkinsMixSingle(lowBits(m1),highBits(m2),lowBits(m3));
  uint32_t hi = jenkinsMixSingle(highBits(m1),lowBits(m2),highBits(m3));
  return combine(hi,lo);
}

uint64_t Hash::simpleHash(const int* input, int len)
{
  uint64_t m1 = 123456789;
  uint64_t m2 = 314159265;
  uint64_t m3 = 958473711;
  uint32_t m4 = 0xCAFEBABEU;
  for(int i = 0; i<len; i++)
  {
    int c = input[i];
    m1 = m1 * 31 + (uint64_t)c;
    m2 = m2 * 317 + (uint64_t)c;
    m3 = m3 * 1609 + (uint64_t)c;
    m4 += (uint32_t)c;
    m4 += (m4 << 10);
    m4 ^= (m4 >> 6);
  }
  m4 += (m4 << 3);
  m4 ^= (m4 >> 11);
  m4 += (m4 << 15);

  uint32_t lo = jenkinsMixSingle(lowBits(m1),highBits(m2),lowBits(m3));
  uint32_t hi = jenkinsMixSingle(highBits(m1),lowBits(m2),highBits(m3));
  return combine(hi ^ m4, lo + m4);
}

//Hash128------------------------------------------------------------------------

ostream& operator<<(ostream& out, const Hash128 other)
{
  out << Global::uint64ToHexString(other.hash1)
      << Global::uint64ToHexString(other.hash0);
  return out;
}

string Hash128::toString() const {
  return Global::uint64ToHexString(hash1) + Global::uint64ToHexString(hash0);
}

Hash128 Hash128::ofString(const string& s) {
  if(s.size() != 32)
    throw IOError("Could not parse as Hash128: " + s);
  for(char c: s) {
    if(!(c >= '0' && c <= '9') &&
       !(c >= 'A' && c <= 'F') &&
       !(c >= 'a' && c <= 'f')
    ) {
      throw IOError("Could not parse as Hash128: " + s);
    }
  }
  uint64_t h1 = Global::hexStringToUInt64(s.substr(0,16));
  uint64_t h0 = Global::hexStringToUInt64(s.substr(16,16));
  return Hash128(h0,h1);
}
