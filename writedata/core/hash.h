/*
 * hash.h
 * Author: David Wu
 */

#ifndef HASH_H_
#define HASH_H_

#include <iostream>
using namespace std;

namespace Hash
{
  //Splitting of uint64s
  uint32_t highBits(uint64_t x);
  uint32_t lowBits(uint64_t x);
  uint64_t combine(uint32_t hi, uint32_t lo);

  uint64_t basicLCong(uint64_t x);
  uint64_t murmurMix(uint64_t x);

  void jenkinsMix(uint32_t& a, uint32_t& b, uint32_t& c);
  uint32_t jenkinsMixSingle(uint32_t a, uint32_t b, uint32_t c);

  uint64_t simpleHash(const char* str);
  uint64_t simpleHash(const int* input, int len);
}

//Hash is "little endian" in the sense that if you printed out hash1, then hash0, you would
//get the standard 128 bit hash string as would be output by something like MD5, as well as
//for comparison.
struct Hash128
{
  uint64_t hash0;
  uint64_t hash1;

  Hash128();
  Hash128(uint64_t hash0, uint64_t hash1);

  bool operator<(const Hash128 other) const;
  bool operator>(const Hash128 other) const;
  bool operator<=(const Hash128 other) const;
  bool operator>=(const Hash128 other) const;
  bool operator==(const Hash128 other) const;
  bool operator!=(const Hash128 other) const;

  friend ostream& operator<<(ostream& out, const Hash128 other);
};

inline Hash128::Hash128()
:hash0(0),hash1(0)
{}

inline Hash128::Hash128(uint64_t hash0, uint64_t hash1)
:hash0(hash0),hash1(hash1)
{}



inline bool Hash128::operator==(const Hash128 other) const
{return hash0 == other.hash0 && hash1 == other.hash1;}

inline bool Hash128::operator!=(const Hash128 other) const
{return hash0 != other.hash0 || hash1 != other.hash1;}

inline bool Hash128::operator>(const Hash128 other) const
{
  if(hash1 > other.hash1) return true;
  if(hash1 < other.hash1) return false;
  return hash0 > other.hash0;
}
inline bool Hash128::operator>=(const Hash128 other) const
{
  if(hash1 > other.hash1) return true;
  if(hash1 < other.hash1) return false;
  return hash0 >= other.hash0;
}
inline bool Hash128::operator<(const Hash128 other) const
{
  if(hash1 < other.hash1) return true;
  if(hash1 > other.hash1) return false;
  return hash0 < other.hash0;
}
inline bool Hash128::operator<=(const Hash128 other) const
{
  if(hash1 < other.hash1) return true;
  if(hash1 > other.hash1) return false;
  return hash0 <= other.hash0;
}


#endif /* HASH_H_ */
