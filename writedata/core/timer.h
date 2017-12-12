/*
 * timer.h
 * Author: David Wu
 *
 * A simple class for getting elapsed runtime. Should be wall time, not cpu time.
 * Should also be threadsafe for concurrent calls to getSeconds()
 *
 */

#ifndef TIMER_H_
#define TIMER_H_

#include <stdint.h>

class ClockTimer
{
  int64_t initialTime;

  public:
  ClockTimer();
  ~ClockTimer();

  void reset();
  double getSeconds();

  //Return some integer indicating the current system time (for seeds/hashes), may vary with OS.
  static int64_t getPrecisionSystemTime();
};


#endif /* TIMER_H_ */
