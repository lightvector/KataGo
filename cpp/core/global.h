/*
 * global.h
 * Author: David Wu
 *
 * Various generic useful things used throughout the program.
 */

#ifndef CORE_GLOBAL_H_
#define CORE_GLOBAL_H_

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <memory>

//GLOBAL DEFINES AND FLAGS----------------------------------------------------
#ifdef __GNUG__  //On g++ only

#define NORETURN __attribute__ ((noreturn))
#define PUREFUNC __attribute__ ((pure))

#else //On other compilers

#define NORETURN
#define PUREFUNC

#endif

#ifdef NDEBUG
//This is so that we can "assert false" in unreachable code branches even in the
//presence of NDEBUG without the compiler complaining about uniniitalized values.
//Ideally sparingly, since the point of NDEBUG presumably is to avoid unnecessary
//runtime checks, but often this is still convenient.
class asserted_unreachable: public std::exception {
  const char* what() const throw() final {
    return "BUG? Reached asserted-unreachable point of the code!";
  }
};
#define ASSERT_UNREACHABLE (throw asserted_unreachable())
#else
#define ASSERT_UNREACHABLE (assert(false))
#endif

//GLOBAL FUNCTIONS------------------------------------------------------------
namespace Global
{
  //ERRORS----------------------------------

  //Report fatal error message and exit
  void fatalError(const char* s) NORETURN;
  void fatalError(const std::string& s) NORETURN;

  //TIME------------------------------------

  //Get string describing the current date, suitable for filenames
  std::string getDateString();

  //STRINGS---------------------------------

  //To string conversions
  std::string boolToString(bool b);
  std::string charToString(char c);
  std::string intToString(int x);
  std::string floatToString(float x);
  std::string doubleToString(double x);
  std::string doubleToStringHighPrecision(double x);
  std::string int64ToString(int64_t x);
  std::string uint32ToString(uint32_t x);
  std::string uint64ToString(uint64_t x);
  std::string uint32ToHexString(uint32_t x);
  std::string uint64ToHexString(uint64_t x);

  //String to conversions using the standard library parsing
  int stringToInt(const std::string& str);
  int64_t stringToInt64(const std::string& str);
  uint64_t stringToUInt64(const std::string& str);
  uint64_t hexStringToUInt64(const std::string& str);
  float stringToFloat(const std::string& str);
  double stringToDouble(const std::string& str);
  bool stringToBool(const std::string& str);
  bool tryStringToInt(const std::string& str, int& x);
  bool tryStringToInt64(const std::string& str, int64_t& x);
  bool tryStringToUInt64(const std::string& str, uint64_t& x);
  bool tryHexStringToUInt64(const std::string& str, uint64_t& x);
  bool tryStringToFloat(const std::string& str, float& x);
  bool tryStringToDouble(const std::string& str, double& x);
  bool tryStringToBool(const std::string& str, bool& x);

  //Check if string is all whitespace
  bool isWhitespace(char c);
  bool isWhitespace(const std::string& s);

  //Check prefix/suffix
  bool isPrefix(const std::string& s, const std::string& prefix);
  bool isSuffix(const std::string& s, const std::string& suffix);
  std::string chopPrefix(const std::string& s, const std::string& prefix);
  std::string chopSuffix(const std::string& s, const std::string& suffix);

  //Trim whitespace off both ends of string
  std::string trim(const std::string& s, const std::string& delimStr = " \t\r\n\v\f");

  //Join strings with a delimiter between each one, from [start,end)
  std::string concat(const char* const* strs, size_t len, const char* delim);
  std::string concat(const std::vector<std::string>& strs, const char* delim);
  std::string concat(const std::vector<std::string>& strs, const char* delim, std::size_t start, std::size_t end);
  std::string concat(const std::set<std::string>& strs, const char* delim);

  //Split string into tokens, trimming off whitespace
  std::vector<std::string> split(const std::string& s);
  //Split string based on the given delim, no trimming
  std::vector<std::string> split(const std::string& s, char delim);

  //Convert to upper or lower case
  std::string toUpper(const std::string& s);
  std::string toLower(const std::string& s);

  bool isEqualCaseInsensitive(const std::string& s0, const std::string& s1);

  //Like sprintf, but returns a string
  std::string strprintf(const char* fmt, ...);

  //Check if a string consists entirely of digits, and parse the integer, checking for overflow
  bool isDigits(const std::string& str);
  bool isDigits(const std::string& str, size_t start, size_t end);
  int parseDigits(const std::string& str);
  int parseDigits(const std::string& str, size_t start, size_t end);

  //Character properties
  bool isDigit(char c);
  bool isAlpha(char c);

  //Check if every char in the string is in the allowed list
  bool stringCharsAllAllowed(const std::string& str, const char* allowedChars);

  //Strips "#" rest-of-line style comments from a string
  std::string stripComments(const std::string& str);

  //Key value pairs are of the form "x=y" or "x = y".
  //Multiple key value pairs are allowed on one line if comma separated.
  //Key value pairs are also broken by newlines.
  std::map<std::string, std::string> readKeyValues(const std::string& contents);

  //Read a memory value, like 16G or 256K.
  uint64_t readMem(const char* str);
  uint64_t readMem(const std::string& str);

  //Display a message and ask the user to press a key to continue
  void pauseForKey();

  //Round x to the nearest multiple of 1/inverseScale
  double roundStatic(double x, double inverseScale);
  //Round x to this many decimal digits of precision
  double roundDynamic(double x, int precision);

}

struct StringError : public std::exception {
  std::string message;
  StringError(const char* m)
    :exception(),message(m)
  {}
  StringError(const std::string& m)
    :exception(),message(m)
  {}

  const char* what() const throw () final
  {return message.c_str();}
};

//Common exception for IO
struct IOError final : public StringError { IOError(const char* msg):StringError(msg) {}; IOError(const std::string& msg):StringError(msg) {}; };
//Common exception for parameter values
struct ValueError final : public StringError { ValueError(const char* msg):StringError(msg) {}; ValueError(const std::string& msg):StringError(msg) {}; };
//Common exception for command line argument handling
struct CommandError final : public StringError { CommandError(const char* msg):StringError(msg) {}; CommandError(const std::string& msg):StringError(msg) {}; };

//Named pairs and triples of data values
#define STRUCT_NAMED_SINGLE(A,B,C) struct C {A B; inline C(): B() {} inline C(A s_n_p_arg_0): B(s_n_p_arg_0) {}}
#define STRUCT_NAMED_PAIR(A,B,C,D,E) struct E {A B; C D; inline E(): B(),D() {} inline E(A s_n_p_arg_0, C s_n_p_arg_1): B(s_n_p_arg_0),D(s_n_p_arg_1) {}}
#define STRUCT_NAMED_TRIPLE(A,B,C,D,E,F,G) struct G {A B; C D; E F; inline G(): B(),D(),F() {} inline G(A s_n_p_arg_0, C s_n_p_arg_1, E s_n_p_arg_2): B(s_n_p_arg_0),D(s_n_p_arg_1),F(s_n_p_arg_2) {}}
#define STRUCT_NAMED_QUAD(A,B,C,D,E,F,G,H,I) struct I {A B; C D; E F; G H; inline I(): B(),D(),F(),H() {} inline I(A s_n_p_arg_0, C s_n_p_arg_1, E s_n_p_arg_2, G s_n_p_arg_3): B(s_n_p_arg_0),D(s_n_p_arg_1),F(s_n_p_arg_2),H(s_n_p_arg_3) {}}

//SHORTCUTS FOR std::map and other containers------------------------------------------------

bool contains(const char* str, char c);
bool contains(const std::string& str, char c);

template<typename A>
bool contains(const std::vector<A>& vec, const A& elt)
{
  for(const A& x: vec)
    if(x == elt)
      return true;
  return false;
}

bool contains(const std::vector<std::string>& vec, const char* elt);

template<typename A>
size_t indexOf(const std::vector<A>& vec, const A& elt)
{
  size_t size = vec.size();
  for(size_t i = 0; i<size; i++)
    if(vec[i] == elt)
      return i;
  return std::string::npos;
}

size_t indexOf(const std::vector<std::string>& vec, const char* elt);

template<typename A>
bool contains(const std::set<A>& set, const A& elt)
{
  return set.find(elt) != set.end();
}

bool contains(const std::set<std::string>& set, const char* elt);

template<typename A, typename B>
bool contains(const std::map<A,B>& m, const A& key)
{
  return m.find(key) != m.end();
}

template<typename B>
bool contains(const std::map<std::string,B>& m, const char* key)
{
  return m.find(std::string(key)) != m.end();
}

template<typename A, typename B>
B map_get(const std::map<A,B>& m, const A& key)
{
  typename std::map<A,B>::const_iterator it = m.find(key);
  if(it == m.end())
    Global::fatalError("map_get: key not found");
  return it->second;
}

template<typename B>
B map_get(const std::map<std::string,B>& m, const char* key)
{
  typename std::map<std::string,B>::const_iterator it = m.find(std::string(key));
  if(it == m.end())
    Global::fatalError(std::string("map_get: key \"") + std::string(key) + std::string("\" not found"));
  return it->second;
}

template<typename A, typename B>
B map_get_defaulting(const std::map<A,B>& m, const A& key, const B& def)
{
  typename std::map<A,B>::const_iterator it = m.find(key);
  if(it == m.end())
    return def;
  return it->second;
}

using unique_ptr_void = std::unique_ptr<void, void(*)(const void*)>;
template<typename T>
unique_ptr_void make_unique_void(T* ptr)
{
  return unique_ptr_void(ptr, [](const void* data) {
    const T* orig = static_cast<const T*>(data);
    delete orig;
  });
}

template<typename T, typename DeleterRet, DeleterRet (*deleter)(T)>
struct WrappedWithDeleter {
  bool assigned;
  T val;
  WrappedWithDeleter(): assigned(false) {}
  WrappedWithDeleter(const T& v): assigned(true), val(v) {}
  ~WrappedWithDeleter() {
    if(assigned)
      deleter(val);
  }
  operator T&() { return val; }
  operator T() const { return val; }
  WrappedWithDeleter& operator=(const T& v) {
    assigned = true;
    val = v;
    return *this;
  }
};

namespace Global {
  template <typename F>
  struct CustomScopeGuard {
    CustomScopeGuard(const CustomScopeGuard&) = delete;
    CustomScopeGuard(CustomScopeGuard&&) = delete;
    CustomScopeGuard& operator=(const CustomScopeGuard&) = delete;

    CustomScopeGuard(F&& f): func(std::forward<F>(f))
    {}

    ~CustomScopeGuard() {
      func();
    }

  private:
    F func;
  };

}


#endif  // CORE_GLOBAL_H_
