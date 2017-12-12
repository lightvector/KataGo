/*
 * md5.h
 * Author: David Wu
 */

#ifndef MD5_H_
#define MD5_H_

#include <stdint.h>

namespace MD5
{
  void get(uint8_t* initial_msg, size_t initial_len, uint64_t hash[2]);
}


#endif /* MD5_H_ */
