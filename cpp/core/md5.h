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
  void get(uint8_t* initial_msg, std::size_t initial_len, uint64_t hash[2]);
}


#endif  // CORE_MD5_H_
