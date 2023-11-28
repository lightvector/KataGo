#ifndef CORE_DATETIME_H_
#define CORE_DATETIME_H_

#include <cstring>
#include <ctime>
#include <iostream>

namespace DateTime {
  //Get the current time
  time_t getNow();

  //Return a tm struct expressing the gm time or local time
  std::tm gmTime(time_t time);
  std::tm localTime(time_t time);

  //Get a string representation of the date
  std::string getDateString();

  //Get a compact string representation of the date and time usable in filenames
  std::string getCompactDateTimeString();

  //Write the time to out as specified by fmt. See std::put_time docs for the format specifiers.
  void writeTimeToStream(std::ostream& out, const char* fmt, time_t time);

  //For debugging/testing
  void runTests();
}

struct SimpleDate {
  int year;
  int month;
  int day;
  SimpleDate();
  SimpleDate(int y, int m, int d);
  SimpleDate(const SimpleDate& other);
  SimpleDate(const std::string& s);
  SimpleDate& operator=(const SimpleDate& other);
  std::string toString() const;
  int numDaysAfter(const SimpleDate& other) const;
  int numDaysIntoYear() const; // Jan 1 is 0, Jan 2 is 1, etc.
  bool isDuringLeapYear() const;

  bool operator==(const SimpleDate& other) const;
  bool operator!=(const SimpleDate& other) const;
  bool operator<(const SimpleDate& other) const;
  bool operator<=(const SimpleDate& other) const;
  bool operator>(const SimpleDate& other) const;
  bool operator>=(const SimpleDate& other) const;

  SimpleDate& operator+=(int n);
  SimpleDate& operator-=(int n);
  friend SimpleDate operator+(SimpleDate a, int b);
  friend SimpleDate operator+(int a, SimpleDate b);
  friend SimpleDate operator-(SimpleDate a, int b);
};


#endif //CORE_DATETIME_H
