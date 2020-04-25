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
  initialTime = (int64_t)GetTickCount64();
}

double ClockTimer::getSeconds() const
{
  int64_t newTime = (int64_t)GetTickCount64();
  return (double)(newTime-initialTime)/1000.0;
}

int64_t ClockTimer::getPrecisionSystemTime()
{
  return (int64_t)GetTickCount64();
}

#endif

//UNIX IMPLEMENTATION------------------------------------------------------------------

#ifdef OS_IS_UNIX_OR_APPLE
#include <chrono>

ClockTimer::ClockTimer()
{
  reset();
}

ClockTimer::~ClockTimer()
{

}

void ClockTimer::reset()
{
  auto d = std::chrono::steady_clock::now().time_since_epoch();
  initialTime = std::chrono::duration<int64_t,std::nano>(d).count();
}

double ClockTimer::getSeconds() const
{
  auto d = std::chrono::steady_clock::now().time_since_epoch();
  int64_t newTime = std::chrono::duration<int64_t,std::nano>(d).count();
  return (double)(newTime-initialTime) / 1000000000.0;
}

int64_t ClockTimer::getPrecisionSystemTime()
{
  auto d = std::chrono::steady_clock::now().time_since_epoch();
  int64_t newTime = std::chrono::duration<int64_t,std::nano>(d).count();
  return newTime;
}

#endif
