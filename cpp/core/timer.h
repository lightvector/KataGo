/*
 * timer.h
 * Author: David Wu
 *
 * A simple class for getting elapsed runtime. Should be wall time, not cpu time.
 * Should also be threadsafe for concurrent calls to getSeconds()
 *
 */

#ifndef CORE_TIMER_H_
#define CORE_TIMER_H_

#include <stdint.h>

class ClockTimer
{
  int64_t initialTime;

  public:
  ClockTimer();
  ~ClockTimer();

  ClockTimer(const ClockTimer&) = delete;
  ClockTimer& operator=(const ClockTimer&) = delete;

  void reset();
  double getSeconds() const;

  //Return some integer indicating the current system time (for seeds/hashes), may vary with OS.
  static int64_t getPrecisionSystemTime();
};


#endif  // CORE_TIMER_H_
