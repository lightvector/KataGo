/*
 * timer.cpp
 * Author: David Wu
 */

#ifdef _WIN32
 #define _TIMER_IS_WINDOWS
#elif _WIN64
 #define _TIMER_IS_WINDOWS
#elif __unix || __APPLE__
  #define _TIMER_IS_UNIX
#else
 #error Unknown OS!
#endif

#ifdef _TIMER_IS_WINDOWS
  #include <windows.h>
#endif
#ifdef _TIMER_IS_UNIX
  #include <sys/time.h>
#endif

#include <stdint.h>
#include <ctime>
#include "timer.h"
using namespace std;

//WINDOWS IMPLMENTATIION-------------------------------------------------------------

#ifdef _TIMER_IS_WINDOWS

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

#ifdef _TIMER_IS_UNIX

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


