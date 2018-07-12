/*
 * global.cpp
 * Author: David Wu
 */

#include <cstdio>
#include <cstdarg>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cctype>
#include <cstring>
#include <dirent.h>
#include <inttypes.h>
#include <sys/types.h>
#include "global.h"
using namespace std;

//ERRORS----------------------------------

void Global::fatalError(const char* s)
{
  cout << "\nFATAL ERROR:\n" << s << endl;
  exit(EXIT_FAILURE);
}

void Global::fatalError(const string& s)
{
  cout << "\nFATAL ERROR:\n" << s << endl;
  exit(EXIT_FAILURE);
}

//TIME------------------------------------

string Global::getDateString()
{
  time_t rawtime;
  time(&rawtime);
  tm* ptm = gmtime(&rawtime);

  ostringstream out;
  out << (ptm->tm_year+1900) << "-"
      << (ptm->tm_mon+1) << "-"
      << (ptm->tm_mday);
  return out.str();
}

//STRINGS---------------------------------

string Global::boolToString(bool b)
{
  return b ? string("true") : string("false");
}

string Global::charToString(char c)
{
  char buf[2];
  buf[0] = c;
  buf[1] = 0;
  return string(buf);
}

string Global::intToString(int x)
{
  stringstream ss;
  ss << x;
  return ss.str();
}

string Global::doubleToString(double x)
{
  stringstream ss;
  ss << x;
  return ss.str();
}

string Global::int64ToString(int64_t x)
{
  stringstream ss;
  ss << x;
  return ss.str();
}

string Global::uint32ToHexString(uint32_t x)
{
  static const char* digits = "0123456789ABCDEF";
  size_t hex_len = sizeof(uint32_t) * 2;
  string s(hex_len,'0');
  for(size_t i = 0; i < hex_len; i++)
    s[hex_len-i-1] = digits[(x >> (i*4)) & 0x0f];
  return s;
}

string Global::uint64ToHexString(uint64_t x)
{
  static const char* digits = "0123456789ABCDEF";
  size_t hex_len = sizeof(uint64_t) * 2;
  string s(hex_len,'0');
  for(size_t i = 0; i < hex_len; i++)
    s[hex_len-i-1] = digits[(x >> (i*4)) & 0x0f];
  return s;
}

bool Global::tryStringToInt(const string& str, int& x)
{
  int val = 0;
  istringstream in(trim(str));
  in >> val;
  if(in.fail() || in.peek() != EOF)
    return false;
  x = val;
  return true;
}

int Global::stringToInt(const string& str)
{
  int val = 0;
  istringstream in(trim(str));
  in >> val;
  if(in.fail() || in.peek() != EOF)
    throw IOError(string("could not parse int: ") + str);
  return val;
}

bool Global::tryStringToInt64(const string& str, int64_t& x)
{
  int64_t val = 0;
  istringstream in(trim(str));
  in >> val;
  if(in.fail() || in.peek() != EOF)
    return false;
  x = val;
  return true;
}

int64_t Global::stringToInt64(const string& str)
{
  int64_t val = 0;
  istringstream in(trim(str));
  in >> val;
  if(in.fail() || in.peek() != EOF)
    throw IOError(string("could not parse int: ") + str);
  return val;
}

bool Global::tryStringToBool(const string& str, bool& x)
{
  string s = toLower(trim(str));
  if(s == "false")
  {x = false; return true;}
  if(s == "true")
  {x = true; return true;}
  return false;
}

bool Global::stringToBool(const string& str)
{
  string s = toLower(trim(str));
  if(s == "false")
    return false;
  if(s == "true")
    return true;

  throw IOError(string("could not parse bool: ") + str);
  return false;
}

bool Global::tryStringToUInt64(const string& str, uint64_t& x)
{
  uint64_t val = 0;
  istringstream in(trim(str));
  in >> val;
  if(in.fail() || in.peek() != EOF)
  {
    istringstream inhex(trim(str));
    inhex >> hex >> val;
    if(inhex.fail() || inhex.peek() != EOF)
      return false;
    x = val;
    return true;
  }
  x = val;
  return true;
}

uint64_t Global::stringToUInt64(const string& str)
{
  uint64_t val = 0;
  istringstream in(trim(str));
  in >> val;
  if(in.fail() || in.peek() != EOF)
    throw IOError(string("could not parse uint64: ") + str);
  return val;
}

bool Global::tryStringToDouble(const string& str, double& x)
{
  double val = 0;
  istringstream in(trim(str));
  in >> val;
  if(in.fail() || in.peek() != EOF)
    return false;
  x = val;
  return true;
}

double Global::stringToDouble(const string& str)
{
  double val = 0;
  istringstream in(trim(str));
  in >> val;
  if(in.fail() || in.peek() != EOF)
    throw IOError(string("could not parse double: ") + str);
  return val;
}

bool Global::isWhitespace(char c)
{
  return contains(" \t\r\n",c);
}

bool Global::isWhitespace(const string& s)
{
  size_t p = s.find_first_not_of(" \t\r\n");
  return p == string::npos;
}

bool Global::isPrefix(const string& s, const string& prefix)
{
  if(s.length() < prefix.length())
    return false;
  int result = s.compare(0, prefix.length(), prefix);
  return result == 0;
}
bool Global::isSuffix(const string& s, const string& suffix)
{
  if(s.length() < suffix.length())
    return false;
  int result = s.compare(s.length() - suffix.length(), suffix.length(), suffix);
  return result == 0;
}


string Global::trim(const string& s)
{
  size_t p2 = s.find_last_not_of(" \t\r\n");
  if (p2 == string::npos)
    return string();
  size_t p1 = s.find_first_not_of(" \t\r\n");
  if (p1 == string::npos)
    p1 = 0;

  return s.substr(p1,(p2-p1)+1);
}

vector<string> Global::split(const string& s)
{
  istringstream in(s);
  string token;
  vector<string> tokens;
  while(in >> token)
  {
    token = Global::trim(token);
    tokens.push_back(token);
  }
  return tokens;
}

string Global::concat(const char* const* strs, size_t len, const char* delim)
{
  size_t totalLen = 0;
  size_t delimLen = strlen(delim);
  for(size_t i = 0; i<len; i++)
  {
    if(i > 0)
      totalLen += delimLen;
    totalLen += strlen(strs[i]);
  }
  string s;
  s.reserve(totalLen);
  for(size_t i = 0; i<len; i++)
  {
    if(i > 0)
      s += delim;
    s += strs[i];
  }
  return s;
}

string Global::concat(const vector<string>& strs, const char* delim)
{
  return concat(strs,delim,0,strs.size());
}

string Global::concat(const vector<string>& strs, const char* delim, size_t start, size_t end)
{
  size_t totalLen = 0;
  size_t delimLen = strlen(delim);
  for(size_t i = start; i<end; i++)
  {
    if(i > start)
      totalLen += delimLen;
    totalLen += strs[i].size();
  }
  string s;
  s.reserve(totalLen);
  for(size_t i = start; i<end; i++)
  {
    if(i > start)
      s += delim;
    s += strs[i];
  }
  return s;
}

vector<string> Global::split(const string& s, char delim)
{
  istringstream in(s);
  string token;
  vector<string> tokens;
  while(getline(in,token,delim))
    tokens.push_back(token);
  return tokens;
}

string Global::toUpper(const string& s)
{
  string t = s;
  int len = t.length();
  for(int i = 0; i<len; i++)
    t[i] = toupper(t[i]);
  return t;
}

string Global::toLower(const string& s)
{
  string t = s;
  int len = t.length();
  for(int i = 0; i<len; i++)
    t[i] = tolower(t[i]);
  return t;
}

static string vformat (const char *fmt, va_list ap)
{
  // Allocate a buffer on the stack that's big enough for us almost
  // all the time.  Be prepared to allocate dynamically if it doesn't fit.
  size_t size = 4096;
  char stackbuf[size];
  std::vector<char> dynamicbuf;
  char *buf = &stackbuf[0];

  int needed;
  while(true)
  {
    // Try to vsnprintf into our buffer.
    needed = vsnprintf(buf, size, fmt, ap);
    // NB. C99 (which modern Linux and OS X follow) says vsnprintf
    // failure returns the length it would have needed.  But older
    // glibc and current Windows return -1 for failure, i.e., not
    // telling us how much was needed.

    if(needed <= (int)size && needed >= 0)
      break;

    // vsnprintf reported that it wanted to write more characters
    // than we allotted.  So try again using a dynamic buffer.  This
    // doesn't happen very often if we chose our initial size well.
    size = (needed > 0) ? (needed+1) : (size*2);
    dynamicbuf.resize(size+1);
    buf = &dynamicbuf[0];
  }
  return std::string(buf, (size_t)needed);
}

string Global::strprintf(const char* fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  std::string buf = vformat (fmt, ap);
  va_end (ap);
  return buf;
}

bool Global::isDigit(char c)
{
  return c >= '0' && c <= '9';
}

bool Global::isAlpha(char c)
{
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool Global::isDigits(const string& str)
{
  return isDigits(str,0,str.size());
}

bool Global::isDigits(const string& str, int start, int end)
{
  //Too long to fit in integer for sure?
  if(end <= start)
    return false;
  if(end-start > 9)
    return false;

  int size = str.size();
  int64_t value = 0;
  for(int i = start; i<end && i<size; i++)
  {
    char c = str[i];
    if(!isDigit(c))
      return false;
    value = value*10 + (c-'0');
  }

  if((value & 0x7FFFFFFFLL) != value)
    return false;

  return true;
}

int Global::parseDigits(const string& str)
{
  return parseDigits(str,0,str.size());
}

int Global::parseDigits(const string& str, int start, int end)
{
  //Too long to fit in integer for sure?
  if(end <= start)
    throw IOError("Could not parse digits, end <= start, or empty string");
  if(end-start > 9)
    throw IOError("Could not parse digits, overflow: " + str.substr(start,end-start));

  int size = str.size();
  int64_t value = 0;
  for(int i = start; i<end && i<size; i++)
  {
    char c = str[i];
    if(!isDigit(c))
      return 0;
    value = value*10 + (c-'0');
  }

  if((value & 0x7FFFFFFFLL) != value)
    throw IOError("Could not parse digits, overflow: " + str.substr(start,end-start));

  return (int)value;
}

bool contains(const char* str, char c)
{
  return strchr(str,c) != NULL;
}

bool contains(const string& str, char c)
{
  return strchr(str.c_str(),c) != NULL;
}

bool contains(const vector<string>& vec, const char* elt)
{
  for(const string& x: vec)
    if(x == elt)
      return true;
  return false;
}

bool contains(const set<string>& set, const char* elt)
{
  return set.find(elt) != set.end();
}

size_t indexOf(const vector<string>& vec, const char* elt)
{
  size_t size = vec.size();
  for(size_t i = 0; i<size; i++)
    if(vec[i] == elt)
      return i;
  return string::npos;
}

bool Global::stringCharsAllAllowed(const string& str, const char* allowedChars)
{
  for(size_t i = 0; i<str.size(); i++)
  {
    if(!contains(allowedChars,str[i]))
      return false;
  }
  return true;
}

map<string,string> Global::readKeyValues(const string& contents)
{
  istringstream lineIn(contents);
  string line;
  map<string,string> keyValues;
  while(getline(lineIn,line))
  {
    if(line.length() <= 0) continue;
    istringstream commaIn(line);
    string commaChunk;
    while(getline(commaIn,commaChunk,','))
    {
      if(commaChunk.length() <= 0) continue;
      size_t equalsPos = commaChunk.find_first_of('=');
      if(equalsPos == string::npos) continue;
      string leftChunk = Global::trim(commaChunk.substr(0,equalsPos));
      string rightChunk = Global::trim(commaChunk.substr(equalsPos+1));
      if(leftChunk.length() == 0)
        throw IOError("readKeyValues: key value pair without key: " + line);
      if(rightChunk.length() == 0)
        throw IOError("readKeyValues: key value pair without value: " + line);
      if(keyValues.find(leftChunk) != keyValues.end())
        throw IOError("readKeyValues: duplicate key: " + leftChunk);
      keyValues[leftChunk] = rightChunk;
    }
  }
  return keyValues;
}

string Global::stripComments(const string& str)
{
  if(str.find_first_of('#') == string::npos)
    return str;

  //Turn str into a stream so we can go line by line
  istringstream in(str);
  string line;
  string result;

  while(getline(in,line))
  {
    size_t pos = line.find_first_of('#');
    if(pos != string::npos)
      result += line.substr(0,pos);
    else
      result += line;
    result += "\n";
  }
  return result;
}


uint64_t Global::readMem(const string& str)
{
  if(str.size() < 2)
    throw IOError("Global::readMem: Could not parse amount of memory: " + str);

  size_t end = str.size()-1;
  size_t snd = str.size()-2;

  string numericPart;
  int shiftFactor;
  if     (str.find_first_of("K") == end)  {shiftFactor = 10; numericPart = str.substr(0,end); }
  else if(str.find_first_of("KB") == snd) {shiftFactor = 10; numericPart = str.substr(0,snd); }
  else if(str.find_first_of("M") == end)  {shiftFactor = 20; numericPart = str.substr(0,end); }
  else if(str.find_first_of("MB") == snd) {shiftFactor = 20; numericPart = str.substr(0,snd); }
  else if(str.find_first_of("G") == end)  {shiftFactor = 30; numericPart = str.substr(0,end); }
  else if(str.find_first_of("GB") == snd) {shiftFactor = 30; numericPart = str.substr(0,snd); }
  else if(str.find_first_of("T") == end)  {shiftFactor = 40; numericPart = str.substr(0,end); }
  else if(str.find_first_of("TB") == snd) {shiftFactor = 40; numericPart = str.substr(0,snd); }
  else if(str.find_first_of("P") == end)  {shiftFactor = 50; numericPart = str.substr(0,end); }
  else if(str.find_first_of("PB") == snd) {shiftFactor = 50; numericPart = str.substr(0,snd); }
  else if(str.find_first_of("B") == end)  {shiftFactor = 0;  numericPart = str.substr(0,end); }
  else                                    {shiftFactor = 0;  numericPart = str; }

  if(!isDigits(numericPart))
    throw IOError("Global::readMem: Could not parse amount of memory: " + str);
  uint64_t mem = 0;
  istringstream in(numericPart);
  in >> mem;
  if(in.bad())
    throw IOError("Global::readMem: Could not parse amount of memory: " + str);

  for(int i = 0; i<shiftFactor; i++)
  {
    uint64_t newMem = mem << 1;
    if(newMem < mem)
      throw IOError("Global::readMem: Could not parse amount of memory (too large): " + str);
    mem = newMem;
  }
  return mem;
}

uint64_t Global::readMem(const char* str)
{
  return readMem(string(str));
}


//IO-------------------------------------

//Read entire file whole
string Global::readFile(const char* filename)
{
  ifstream ifs(filename);
  if(!ifs.good())
    throw IOError(string("File not found: ") + filename);
  string str((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());
  return str;
}

string Global::readFile(const string& filename)
{
  return readFile(filename.c_str());
}

//Read file into separate lines, using the specified delimiter character(s).
//The delimiter characters are NOT included.
vector<string> Global::readFileLines(const char* filename, char delimiter)
{
  ifstream ifs(filename);
  if(!ifs.good())
    throw IOError(string("File not found: ") + filename);

  vector<string> vec;
  string line;
  while(getline(ifs,line,delimiter))
    vec.push_back(line);
  return vec;
}

vector<string> Global::readFileLines(const string& filename, char delimiter)
{
  return readFileLines(filename.c_str(), delimiter);
}

void Global::collectFiles(const string& dirname, std::function<bool(const string&)> fileFilter, vector<string>& collected)
{
  DIR *dir;
  struct dirent *entry;

  if(!(dir = opendir(dirname.c_str()))) {
    cerr << "Error (" << errno << ") opening " << dirname << endl;
    return;
  }

  while ((entry = readdir(dir)) != NULL) {
    if(entry->d_type == DT_DIR) {
      if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
        continue;
      const string subPath = dirname + "/" + entry->d_name;
      collectFiles(subPath,fileFilter,collected);
    }
    else {
      string fileName = string(entry->d_name);
      if(fileFilter(fileName))
        collected.push_back(dirname + "/" + fileName);
    }
  }
  closedir(dir);
}


//USER IO----------------------------

void Global::pauseForKey()
{
  cout << "Press any key to continue..." << endl;
  cin.get();
}








