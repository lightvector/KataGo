#include "../core/datetime.h"

#include <chrono>
#include <iomanip>

#include "../core/os.h"
#include "../core/multithread.h"

time_t DateTime::getNow() {
  time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  return time;
}

std::tm DateTime::localTime(time_t time) {
  std::tm buf{};
#if defined(OS_IS_UNIX_OR_APPLE)
  localtime_r(&time, &buf);
#elif defined(OS_IS_WINDOWS)
  localtime_s(&buf, &time);
#else
  static std::mutex localTimeMutex;
  std::lock_guard<std::mutex> lock(localTimeMutex);
  buf = *(std::localtime(&time));
#endif
  return buf;
}

std::string DateTime::getCompactDateTimeString() {
  time_t time = getNow();
  std::tm tm = localTime(time);
  std::ostringstream out;
  out << std::put_time(&tm, "%Y%m%d-%H%M%S");
  return out.str();
}

void DateTime::writeTimeToStream(std::ostream& out, const char* fmt, time_t time) {
  std::tm tm = localTime(time);
  out << std::put_time(&tm, fmt);
}
