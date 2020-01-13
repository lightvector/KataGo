/*
 * md5.h
 * Author: David Wu
 */

#ifndef CORE_MD5_H_
#define CORE_MD5_H_

#include <stdint.h>
#include <cstdlib>

namespace MD5
{
  void get(const char* initial_msg, std::size_t initial_len, uint32_t hash[4]);
  void get(const uint8_t* initial_msg, std::size_t initial_len, uint32_t hash[4]);
}


#endif  // CORE_MD5_H_
