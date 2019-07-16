#ifndef CORE_DATETIME_H_
#define CORE_DATETIME_H_

#include <cstring>
#include <ctime>
#include <iostream>

namespace DateTime {
  //Get the current time
  time_t getNow();

  //Return a tm struct expressing the local time
  std::tm localTime(time_t time);

  //Get a compact string representation of the date and time usable in filenames
  std::string getCompactDateTimeString();

  //Write the time to out as specified by fmt. See std::put_time docs for the format specifiers.
  void writeTimeToStream(std::ostream& out, const char* fmt, time_t time);
}


#endif //CORE_DATETIME_H
