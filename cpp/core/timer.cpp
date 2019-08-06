#include "../core/timer.h"
#include "../core/os.h"

/*
 * timer.cpp
 * Author: David Wu
 */

//WINDOWS IMPLMENTATIION-------------------------------------------------------------

#ifdef OS_IS_WINDOWS
#include <windows.h>
#include <ctime>

ClockTimer::ClockTimer()
{
  reset();
}

ClockTimer::~ClockTimer()
{

}

void ClockTimer::reset()
{
  initialTime = (int64_t)GetTickCount();
}

double ClockTimer::getSeconds() const
{
  int64_t newTime = (int64_t)GetTickCount();
  return (double)(newTime-initialTime)/1000.0;
}

int64_t ClockTimer::getPrecisionSystemTime()
{
  return (int64_t)GetTickCount();
}

#endif

//UNIX IMPLEMENTATION------------------------------------------------------------------

#ifdef OS_IS_UNIX_OR_APPLE
#include <sys/time.h>
#include <ctime>

ClockTimer::ClockTimer()
{
  reset();
}

ClockTimer::~ClockTimer()
{

}

void ClockTimer::reset()
{
  struct timeval timeval;
  gettimeofday(&timeval,NULL);
  initialTime = (int64_t)timeval.tv_sec * 1000000LL + (int64_t)timeval.tv_usec;
}

double ClockTimer::getSeconds() const
{
  struct timeval timeval;
  gettimeofday(&timeval,NULL);
  int64_t newTime = (int64_t)timeval.tv_sec * 1000000LL + (int64_t)timeval.tv_usec;
  return (double)(newTime-initialTime)/1000000.0;
}

int64_t ClockTimer::getPrecisionSystemTime()
{
  struct timeval timeval;
  gettimeofday(&timeval,NULL);
  return (int64_t)timeval.tv_sec * 1000000LL + (int64_t)timeval.tv_usec;
}

#endif
