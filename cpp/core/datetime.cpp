#include "../core/datetime.h"

#include <chrono>
#include <iomanip>
#include <sstream>

#include "../core/os.h"
#include "../core/multithread.h"
#include "../core/rand.h"
#include "../core/test.h"

#ifdef __MINGW32__
#include <timezoneapi.h>
#endif

/* MinGW doesn't support date specifier arguments correctly.
 * It skips `%F`, `%T` and prints `W. Europe Summer Time` instead of `+0200` for the timezone `%z` specifier.
 * The function also handles DST (Daylight saving time).
 * So, create the necessary format manually.
 */
const char* DateTime::getTimeFormat() {
#ifdef __MINGW32__
  // Get current time to check if we're in DST
  std::time_t now = std::time(nullptr);
  std::tm local_tm = *std::localtime(&now);

  // Windows bias is opposite of what we want (negative for east, positive for west)
  // We invert it to match the conventional format (positive for east, negative for west)
  TIME_ZONE_INFORMATION tzi;
  GetTimeZoneInformation(&tzi);
  int total_bias = -(tzi.Bias + (local_tm.tm_isdst > 0 ? tzi.DaylightBias : tzi.StandardBias));

  int hours = abs(total_bias) / 60;
  int minutes = abs(total_bias) % 60;

  static char buffer[28];
  snprintf(buffer, sizeof(buffer), "%%Y-%%m-%%d %%H:%%M:%%S%c%02d%02d: ",
           total_bias >= 0 ? '+' : '-', hours, minutes);
  return buffer;
#else
  return "%F %T%z: ";
#endif
}

time_t DateTime::getNow() {
  time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  return time;
}

std::tm DateTime::gmTime(time_t time) {
  std::tm buf{};
#if defined(OS_IS_UNIX_OR_APPLE)
  gmtime_r(&time, &buf);
#elif defined(OS_IS_WINDOWS)
  gmtime_s(&buf, &time);
#else
  static std::mutex gmTimeMutex;
  std::lock_guard<std::mutex> lock(gmTimeMutex);
  buf = *(std::gmtime(&time));
#endif
  return buf;
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

std::string DateTime::getDateString()
{
  time_t time = getNow();
  std::tm ptm = gmTime(time);
  std::ostringstream out;
  out << (ptm.tm_year+1900) << "-"
      << (ptm.tm_mon+1) << "-"
      << (ptm.tm_mday);
  return out.str();
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

//---------------------------------------------------------------------------------

static bool isLeapYear(int year) {
  return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}
static int numDaysInMonth(int year, int month) {
  if(month == 2)
    return isLeapYear(year) ? 29 : 28;
  else if(month == 4 || month == 6 || month == 9 || month == 11)
    return 30;
  else
    return 31;
}

static bool isValid(int year, int month, int day) {
  if(month < 1 || month > 12 || day < 1 || day > numDaysInMonth(year,month))
    return false;
  return true;
}

SimpleDate::SimpleDate()
  :year(1970), month(1), day(1) {}

SimpleDate::SimpleDate(int y, int m, int d)
  :year(y), month(m), day(d)
{
  if(!isValid(year,month,day))
    throw StringError(
      "SimpleDate: Invalid year month day:"
      + Global::intToString(year) + ","
      + Global::intToString(month) + ","
      + Global::intToString(day)
    );
}

SimpleDate::SimpleDate(const SimpleDate& other)
  :year(other.year), month(other.month), day(other.day)
{}

SimpleDate& SimpleDate::operator=(const SimpleDate& other) {
  year = other.year;
  month = other.month;
  day = other.day;
  return *this;
}

std::string SimpleDate::toString() const {
  char buf[32];
  sprintf(buf, "%04d-%02d-%02d", year, month, day);
  return std::string(buf);
}

SimpleDate::SimpleDate(const std::string& s) {
  if(s.size() != 10 || s[4] != '-' || s[7] != '-')
    throw StringError("SimpleDate: Unable to parse as ISO8601 date: " + s);
  for(int i = 0; i<10; i++) {
    if(i != 4 && i != 7) {
      if(s[i] < '0' || s[i] > '9')
        throw StringError("SimpleDate: Unable to parse as ISO8601 date: " + s);
    }
  }

  std::string yearStr = s.substr(0,4);
  std::string monthStr = s.substr(5,2);
  std::string dayStr = s.substr(8,2);

  year = Global::stringToInt(yearStr);
  month = Global::stringToInt(monthStr);
  day = Global::stringToInt(dayStr);

  if(!isValid(year,month,day))
    throw StringError("SimpleDate: Unable to parse as ISO8601 date: " + s);
}

// Division rounding towards negative infinity, assuming d is positive but n might not be.
static int divFloor(int n, int d) {
  return n >= 0 ? n/d : -1 - (-1-n)/d;
}

static int numLeapYearsUpToAndIncluding(int year) {
  return divFloor(year,4) - divFloor(year,100) + divFloor(year,400);
}
static const int CUMULATIVE_DAYS_UNTIL_MONTH[14] = {
  0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365
};
static const int CUMULATIVE_DAYS_UNTIL_MONTH_LEAP_YEAR[14] = {
  0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366
};

bool SimpleDate::isDuringLeapYear() const {
  return isLeapYear(year);
}

int SimpleDate::numDaysIntoYear() const {
  assert(month >= 1 && month <= 12);
  if(isLeapYear(year))
    return CUMULATIVE_DAYS_UNTIL_MONTH_LEAP_YEAR[month] + (day-1);
  else
    return CUMULATIVE_DAYS_UNTIL_MONTH[month] + (day-1);
}

int SimpleDate::numDaysAfter(const SimpleDate& other) const {
  bool flip = false;
  if(
    year < other.year ||
    (year == other.year && month < other.month) ||
    (year == other.year && month == other.month && day < other.day)
  ) {
    flip = true;
  }

  const SimpleDate& dLater = flip ? other : *this;
  const SimpleDate& dEarly = flip ? *this : other;

  int dayCount = 0;
  // Start by counting days difference between Jan 01 of the respective years.
  dayCount += 365 * (dLater.year - dEarly.year);
  dayCount += numLeapYearsUpToAndIncluding(dLater.year-1) - numLeapYearsUpToAndIncluding(dEarly.year-1);

  assert(dEarly.month >= 1 && dEarly.month <= 12);
  assert(dLater.month >= 1 && dLater.month <= 12);

  // Now adjust for months and days
  dayCount -= dEarly.numDaysIntoYear();
  dayCount += dLater.numDaysIntoYear();
  return flip ? -dayCount : dayCount;
}

bool SimpleDate::operator==(const SimpleDate& other) const {
  return year == other.year && month == other.month && day == other.day;
}
bool SimpleDate::operator!=(const SimpleDate& other) const {
  return !(*this == other);
}
bool SimpleDate::operator<(const SimpleDate& other) const {
  return (
    year < other.year ||
    (year == other.year && month < other.month) ||
    (year == other.year && month == other.month && day < other.day)
  );
}
bool SimpleDate::operator<=(const SimpleDate& other) const {
  return (
    year < other.year ||
    (year == other.year && month < other.month) ||
    (year == other.year && month == other.month && day <= other.day)
  );
}
bool SimpleDate::operator>(const SimpleDate& other) const {
  return !(*this <= other);
}
bool SimpleDate::operator>=(const SimpleDate& other) const {
  return !(*this < other);
}

SimpleDate& SimpleDate::operator+=(int n) {
  // Normalize to Jan 1 while we deal with years.
  n += numDaysIntoYear();
  month = 1;
  day = 1;

  if(n < 0) {
    // Add one, and divide by slightly too small because of leap years - we should overshoot.
    int approxYearsToSub = 1 + (-n) / 365;
    SimpleDate date2 = *this;
    date2.year -= approxYearsToSub;
    n += this->numDaysAfter(date2);
    year -= approxYearsToSub;
    // Since we overshooted, this should be true.
    assert(n >= 0);
  }
  while(n >= 366) {
    // Divide by 366 for leap years, because of non leap years we should undershoot a bit.
    int approxYearsToAdd = n / 366;
    SimpleDate date2 = *this;
    date2.year += approxYearsToAdd;
    n -= date2.numDaysAfter(*this);
    year += approxYearsToAdd;
    // Since we overshooted, this should be true.
    assert(n >= 0);
  }
  if(n == 365) {
    if(isLeapYear(year)) {
      month = 12;
      day = 31;
      return *this;
    }
    else {
      year += 1;
      // month = 1;  // already set
      // day = 1; // already set
      return *this;
    }
  }
  // Now we should be definitely on the right year, with n from 0 to 364 inclusive.
  assert(n >= 0 && n <= 364);
  // Work out days and months.
  if(isLeapYear(year)) {
    while(n >= CUMULATIVE_DAYS_UNTIL_MONTH_LEAP_YEAR[month+1])
      month += 1;
    day = 1 + n - CUMULATIVE_DAYS_UNTIL_MONTH_LEAP_YEAR[month];
  }
  else {
    while(n >= CUMULATIVE_DAYS_UNTIL_MONTH[month+1])
      month += 1;
    day = 1 + n - CUMULATIVE_DAYS_UNTIL_MONTH[month];
  }
  return *this;
}

SimpleDate& SimpleDate::operator-=(int n) {
  return ((*this) += (-n));
}

SimpleDate operator+(SimpleDate a, int b) {
  a += b;
  return a;
}
SimpleDate operator+(int a, SimpleDate b) {
  b += a;
  return b;
}
SimpleDate operator-(SimpleDate a, int b) {
  a -= b;
  return a;
}

void DateTime::runTests() {
  testAssert(SimpleDate(1999,1,11) == SimpleDate(1999,1,11));
  testAssert(SimpleDate(1999,1,11) != SimpleDate(1999,2,11));
  testAssert(SimpleDate(1999,1,11) != SimpleDate(1999,1,21));
  testAssert(SimpleDate(1999,1,11) != SimpleDate(1998,1,11));
  testAssert(!(SimpleDate(1999,1,11) != SimpleDate(1999,1,11)));
  testAssert(!(SimpleDate(1999,1,11) == SimpleDate(1999,2,11)));
  testAssert(!(SimpleDate(1999,1,11) == SimpleDate(1999,1,21)));
  testAssert(!(SimpleDate(1999,1,11) == SimpleDate(1998,1,11)));

  testAssert(SimpleDate(1965,6,21) > SimpleDate(1945,8,25));
  testAssert(SimpleDate(1965,6,21) > SimpleDate(1965,4,25));
  testAssert(SimpleDate(1965,6,21) > SimpleDate(1965,6,20));
  testAssert(SimpleDate(1965,6,21) >= SimpleDate(1945,8,25));
  testAssert(SimpleDate(1965,6,21) >= SimpleDate(1965,4,25));
  testAssert(SimpleDate(1965,6,21) >= SimpleDate(1965,6,20));
  testAssert(SimpleDate(1965,6,21) >= SimpleDate(1965,6,21));

  testAssert(!(SimpleDate(1965,6,21) <= SimpleDate(1945,8,25)));
  testAssert(!(SimpleDate(1965,6,21) <= SimpleDate(1965,4,25)));
  testAssert(!(SimpleDate(1965,6,21) <= SimpleDate(1965,6,20)));
  testAssert(!(SimpleDate(1965,6,21) < SimpleDate(1945,8,25)));
  testAssert(!(SimpleDate(1965,6,21) < SimpleDate(1965,4,25)));
  testAssert(!(SimpleDate(1965,6,21) < SimpleDate(1965,6,20)));
  testAssert(!(SimpleDate(1965,6,21) < SimpleDate(1965,6,21)));

  testAssert(SimpleDate(1965,6,21) < SimpleDate(1985,4,20));
  testAssert(SimpleDate(1965,6,21) < SimpleDate(1965,8,20));
  testAssert(SimpleDate(1965,6,21) < SimpleDate(1965,6,22));
  testAssert(SimpleDate(1965,6,21) <= SimpleDate(1985,4,20));
  testAssert(SimpleDate(1965,6,21) <= SimpleDate(1965,8,20));
  testAssert(SimpleDate(1965,6,21) <= SimpleDate(1965,6,22));
  testAssert(SimpleDate(1965,6,21) <= SimpleDate(1965,6,21));

  testAssert(!(SimpleDate(1965,6,21) >= SimpleDate(1985,4,20)));
  testAssert(!(SimpleDate(1965,6,21) >= SimpleDate(1965,8,20)));
  testAssert(!(SimpleDate(1965,6,21) >= SimpleDate(1965,6,22)));
  testAssert(!(SimpleDate(1965,6,21) > SimpleDate(1985,4,20)));
  testAssert(!(SimpleDate(1965,6,21) > SimpleDate(1965,8,20)));
  testAssert(!(SimpleDate(1965,6,21) > SimpleDate(1965,6,22)));
  testAssert(!(SimpleDate(1965,6,21) > SimpleDate(1965,6,21)));

  testAssert(SimpleDate(1234,5,7) == SimpleDate("1234-05-07"));
  testAssert(SimpleDate() == SimpleDate("1970-01-01"));
  testAssert(SimpleDate(475,1,31) == SimpleDate("0475-01-31"));

  testAssert(SimpleDate(1234,5,7).toString() == "1234-05-07");
  testAssert(SimpleDate().toString() == "1970-01-01");

  testAssert(SimpleDate() + 365 == SimpleDate("1971-01-01"));
  testAssert(SimpleDate() + 365*2 == SimpleDate("1972-01-01"));
  testAssert(SimpleDate() + 365*3 == SimpleDate("1972-12-31"));

  testAssert(SimpleDate("1895-01-01") + 365 == SimpleDate("1896-01-01"));
  testAssert(SimpleDate("1895-01-01") + 365*2 == SimpleDate("1896-12-31"));
  testAssert(SimpleDate("1895-01-01") + 365*3 == SimpleDate("1897-12-31"));

  testAssert(SimpleDate("1899-01-01") + 365 == SimpleDate("1900-01-01"));
  testAssert(SimpleDate("1899-01-01") + 365*2 == SimpleDate("1901-01-01"));
  testAssert(SimpleDate("1899-01-01") + 365*3 == SimpleDate("1902-01-01"));

  testAssert(SimpleDate("1999-01-01") + 365 == SimpleDate("2000-01-01"));
  testAssert(SimpleDate("1999-01-01") + 365*2 == SimpleDate("2000-12-31"));
  testAssert(SimpleDate("1999-01-01") + 365*3 == SimpleDate("2001-12-31"));

  std::ostringstream out;
  Rand rand("SimpleDate tests");
  for(int i = 0; i<100000; i++) {
    SimpleDate date1;
    SimpleDate date2;
    int d1 = rand.nextInt(-10000000,10000000);
    int d2 = rand.nextInt(-20,20);
    d2 *= (rand.nextBool(0.1) ? 100 : 1);
    int d3 = rand.nextInt(-20,20);
    d3 = d3 * d3 * (rand.nextBool(0.5) ? -1 : 1);
    d3 *= (rand.nextBool(0.1) ? 37 : 1);

    date1 += d1;
    testAssert(SimpleDate() + d1 == date1);
    testAssert(d1 + SimpleDate() == date1);
    testAssert(SimpleDate() - (-d1) == date1);
    testAssert(date1 - d1 == SimpleDate());
    testAssert(date1.numDaysAfter(SimpleDate()) == d1);

    date2 = date1;
    date2 += d2;
    testAssert(date1 + d2 == date2);
    testAssert(d2 + date1 == date2);
    testAssert(date1 - (-d2) == date2);
    testAssert(date2 - d2 == date1);
    testAssert(date2.numDaysAfter(date1) == d2);

    testAssert((date1 == SimpleDate()) == (d1 == 0));
    testAssert((date1 != SimpleDate()) == (d1 != 0));
    testAssert((date1 >  SimpleDate()) == (d1 >  0));
    testAssert((date1 >= SimpleDate()) == (d1 >= 0));
    testAssert((date1 <  SimpleDate()) == (d1 <  0));
    testAssert((date1 <= SimpleDate()) == (d1 <= 0));

    testAssert((date2 == date1) == (d2 == 0));
    testAssert((date2 != date1) == (d2 != 0));
    testAssert((date2 >  date1) == (d2 >  0));
    testAssert((date2 >= date1) == (d2 >= 0));
    testAssert((date2 <  date1) == (d2 <  0));
    testAssert((date2 <= date1) == (d2 <= 0));

    SimpleDate date3(date2);
    date3 += d3;
    testAssert(date3.numDaysAfter(date2) == d3);
    testAssert((date3 == date2) == (d3 == 0));
    testAssert((date3 != date2) == (d3 != 0));
    testAssert((date3 >  date2) == (d3 >  0));
    testAssert((date3 >= date2) == (d3 >= 0));
    testAssert((date3 <  date2) == (d3 <  0));
    testAssert((date3 <= date2) == (d3 <= 0));

    testAssert((date1 + d2) + d3 == date3);
    testAssert((date1 + d3) + d2 == date3);
    testAssert(date1 + (d2 + d3) == date3);

    testAssert(
      ((((((SimpleDate()+d1)+d2)-d1)+d3)-d2)-d3) == SimpleDate()
    );

    if(i < 30) {
      out << date1.toString() << " " << date2.toString() << " " << date3.toString() << "\n";
    }
  }

  std::string expected = R"%%(
18575-09-30 18575-09-11 18575-08-26
7737-10-26 7737-10-19 7738-05-03
4389-09-22 4389-09-04 4389-06-15
18885-07-07 18885-06-22 18884-08-02
22422-10-22 22422-10-29 22423-07-12
16796-06-24 16801-09-06 16800-10-17
-20113-05-26 -20113-05-24 -20113-04-18
26298-05-26 26298-05-06 26298-12-17
15912-09-21 15912-09-19 15913-08-09
-1210-08-13 -1210-08-31 -1210-05-02
-16392-02-05 -16393-10-28 -16393-04-15
11800-03-09 11800-02-24 11800-12-10
-3367-10-11 -3370-10-07 -3369-07-23
16152-07-21 16152-07-02 16152-05-27
24875-12-08 24875-11-22 24874-11-26
-15652-12-03 -15652-12-11 -15652-08-12
-1307-03-07 -1307-02-19 -1308-04-01
1665-09-27 1665-10-08 1666-10-04
26656-04-22 26656-04-25 26655-04-30
-1636-12-02 -1636-11-19 -1636-06-28
-20130-02-05 -20130-01-28 -20130-01-12
-13400-08-26 -13400-08-24 -13399-03-08
2597-04-23 2597-04-19 2596-09-06
-4344-11-03 -4347-04-13 -4347-04-17
13710-09-14 13710-09-07 13709-12-25
-17595-04-26 -17595-04-19 -17595-12-31
0079-08-16 0079-08-01 0079-06-13
6794-10-12 6794-09-28 6794-01-15
21022-04-17 21022-04-09 21021-10-22
15578-06-22 15578-07-03 15578-07-02
)%%";

  TestCommon::expect("Sample Random Dates",out,expected);

  for(int i = 0; i<370; i++) {
    if(i == 80)
      i = 340;
    out << (SimpleDate("2011-12-20")+i).toString() << " " << (SimpleDate("2012-12-20")+i).toString() << "\n";
  }

  expected = R"%%(
2011-12-20 2012-12-20
2011-12-21 2012-12-21
2011-12-22 2012-12-22
2011-12-23 2012-12-23
2011-12-24 2012-12-24
2011-12-25 2012-12-25
2011-12-26 2012-12-26
2011-12-27 2012-12-27
2011-12-28 2012-12-28
2011-12-29 2012-12-29
2011-12-30 2012-12-30
2011-12-31 2012-12-31
2012-01-01 2013-01-01
2012-01-02 2013-01-02
2012-01-03 2013-01-03
2012-01-04 2013-01-04
2012-01-05 2013-01-05
2012-01-06 2013-01-06
2012-01-07 2013-01-07
2012-01-08 2013-01-08
2012-01-09 2013-01-09
2012-01-10 2013-01-10
2012-01-11 2013-01-11
2012-01-12 2013-01-12
2012-01-13 2013-01-13
2012-01-14 2013-01-14
2012-01-15 2013-01-15
2012-01-16 2013-01-16
2012-01-17 2013-01-17
2012-01-18 2013-01-18
2012-01-19 2013-01-19
2012-01-20 2013-01-20
2012-01-21 2013-01-21
2012-01-22 2013-01-22
2012-01-23 2013-01-23
2012-01-24 2013-01-24
2012-01-25 2013-01-25
2012-01-26 2013-01-26
2012-01-27 2013-01-27
2012-01-28 2013-01-28
2012-01-29 2013-01-29
2012-01-30 2013-01-30
2012-01-31 2013-01-31
2012-02-01 2013-02-01
2012-02-02 2013-02-02
2012-02-03 2013-02-03
2012-02-04 2013-02-04
2012-02-05 2013-02-05
2012-02-06 2013-02-06
2012-02-07 2013-02-07
2012-02-08 2013-02-08
2012-02-09 2013-02-09
2012-02-10 2013-02-10
2012-02-11 2013-02-11
2012-02-12 2013-02-12
2012-02-13 2013-02-13
2012-02-14 2013-02-14
2012-02-15 2013-02-15
2012-02-16 2013-02-16
2012-02-17 2013-02-17
2012-02-18 2013-02-18
2012-02-19 2013-02-19
2012-02-20 2013-02-20
2012-02-21 2013-02-21
2012-02-22 2013-02-22
2012-02-23 2013-02-23
2012-02-24 2013-02-24
2012-02-25 2013-02-25
2012-02-26 2013-02-26
2012-02-27 2013-02-27
2012-02-28 2013-02-28
2012-02-29 2013-03-01
2012-03-01 2013-03-02
2012-03-02 2013-03-03
2012-03-03 2013-03-04
2012-03-04 2013-03-05
2012-03-05 2013-03-06
2012-03-06 2013-03-07
2012-03-07 2013-03-08
2012-03-08 2013-03-09
2012-11-24 2013-11-25
2012-11-25 2013-11-26
2012-11-26 2013-11-27
2012-11-27 2013-11-28
2012-11-28 2013-11-29
2012-11-29 2013-11-30
2012-11-30 2013-12-01
2012-12-01 2013-12-02
2012-12-02 2013-12-03
2012-12-03 2013-12-04
2012-12-04 2013-12-05
2012-12-05 2013-12-06
2012-12-06 2013-12-07
2012-12-07 2013-12-08
2012-12-08 2013-12-09
2012-12-09 2013-12-10
2012-12-10 2013-12-11
2012-12-11 2013-12-12
2012-12-12 2013-12-13
2012-12-13 2013-12-14
2012-12-14 2013-12-15
2012-12-15 2013-12-16
2012-12-16 2013-12-17
2012-12-17 2013-12-18
2012-12-18 2013-12-19
2012-12-19 2013-12-20
2012-12-20 2013-12-21
2012-12-21 2013-12-22
2012-12-22 2013-12-23
2012-12-23 2013-12-24
)%%";

  TestCommon::expect("Counting forward dates",out,expected);
}


