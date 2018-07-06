/*
 * sha2.h
 * Author: David Wu
 */

#ifndef SHA2_H_
#define SHA2_H_

#include <stdint.h>
#include <cstdlib>

namespace SHA2
{
  //All outputs are "big-endian" in the sense that if you sequentially
  //walked through the hash and wrote it in hex, you would get the standard
  //digest. (within each individual hash array elt, of course, it's still
  //dependent on the architecture).
  //For char output, it's precisely the digest string, with a null terminator.
  //All inputs are also "big-endian" in the same way.
  void get256(const char* msg, char hash[65]);
  void get256(const char* msg, uint8_t hash[32]);
  void get256(const char* msg, uint32_t hash[8]);
  void get256(const char* msg, uint64_t hash[4]);
  void get256(const uint8_t* msg, size_t len, char hash[65]);
  void get256(const uint8_t* msg, size_t len, uint8_t hash[32]);
  void get256(const uint8_t* msg, size_t len, uint32_t hash[8]);
  void get256(const uint8_t* msg, size_t len, uint64_t hash[4]);
  void get256(const uint32_t* msg, size_t len, char hash[65]);
  void get256(const uint32_t* msg, size_t len, uint8_t hash[32]);
  void get256(const uint32_t* msg, size_t len, uint32_t hash[8]);
  void get256(const uint32_t* msg, size_t len, uint64_t hash[4]);

  void get384(const char* msg, char hash[97]);
  void get384(const char* msg, uint8_t hash[48]);
  void get384(const char* msg, uint32_t hash[12]);
  void get384(const char* msg, uint64_t hash[6]);
  void get384(const uint8_t* msg, size_t len, char hash[97]);
  void get384(const uint8_t* msg, size_t len, uint8_t hash[48]);
  void get384(const uint8_t* msg, size_t len, uint32_t hash[12]);
  void get384(const uint8_t* msg, size_t len, uint64_t hash[6]);
  void get384(const uint32_t* msg, size_t len, char hash[97]);
  void get384(const uint32_t* msg, size_t len, uint8_t hash[48]);
  void get384(const uint32_t* msg, size_t len, uint32_t hash[12]);
  void get384(const uint32_t* msg, size_t len, uint64_t hash[6]);

  void get512(const char* msg, char hash[129]);
  void get512(const char* msg, uint8_t hash[64]);
  void get512(const char* msg, uint32_t hash[16]);
  void get512(const char* msg, uint64_t hash[8]);
  void get512(const uint8_t* msg, size_t len, char hash[129]);
  void get512(const uint8_t* msg, size_t len, uint8_t hash[64]);
  void get512(const uint8_t* msg, size_t len, uint32_t hash[16]);
  void get512(const uint8_t* msg, size_t len, uint64_t hash[8]);
  void get512(const uint32_t* msg, size_t len, char hash[129]);
  void get512(const uint32_t* msg, size_t len, uint8_t hash[64]);
  void get512(const uint32_t* msg, size_t len, uint32_t hash[16]);
  void get512(const uint32_t* msg, size_t len, uint64_t hash[8]);
}

#endif /* SHA2_H_ */

