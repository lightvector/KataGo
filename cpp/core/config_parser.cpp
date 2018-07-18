
#include <cmath>
#include <fstream>
#include "../core/config_parser.h"

ConfigParser::ConfigParser(const string& fname)
  :fileName(fname),keyValues()
{
  ifstream in(fileName);
  if(!in.is_open())
    throw IOError("Could not open config file: " + fileName);

  int lineNum = 0;
  string line;
  while(getline(in,line)) {
    lineNum += 1;
    line = Global::trim(line);
    if(line.length() <= 0 || line[0] == '#')
      continue;

    size_t commentPos = line.find("#");
    if(commentPos != string::npos)
      line = line.substr(0, commentPos);

    size_t pos = line.find("=");
    if(pos == string::npos)
      throw IOError("Could not parse kv pair, line " + Global::intToString(lineNum) + " does not have a non-commented '=' in " + fileName);

    string key = Global::trim(line.substr(0,pos));
    string value = Global::trim(line.substr(pos+1));
    keyValues[key] = value;
  }
}

ConfigParser::~ConfigParser()
{}

bool ConfigParser::contains(const string& key) const {
  return keyValues.find(key) != keyValues.end();
}

string ConfigParser::getString(const string& key) const {
  auto iter = keyValues.find(key);
  if(iter == keyValues.end())
    throw IOError("Could not find key '" + key + "' in config file " + fileName);
  return iter->second;
}

string ConfigParser::getString(const string& key, const set<string>& possibles) const {
  auto iter = keyValues.find(key);
  if(iter == keyValues.end())
    throw IOError("Could not find key '" + key + "' in config file " + fileName);
  if(possibles.find(key) == possibles.end())
    throw IOError("Key '" + key + "' must be one of (" + Global::concat(possibles,"|") + ") in config file " + fileName);
  return iter->second;
}

bool ConfigParser::getBool(const string& key) const {
  string value = getString(key);
  bool x;
  if(!Global::tryStringToBool(value,x))
    throw new IOError("Could not parse '" + value + "' as bool for key '" + key + "' in config file " + fileName);
  return x;
}

int ConfigParser::getInt(const string& key) const {
  string value = getString(key);
  int x;
  if(!Global::tryStringToInt(value,x))
    throw new IOError("Could not parse '" + value + "' as int for key '" + key + "' in config file " + fileName);
  return x;
}
int ConfigParser::getInt(const string& key, int min, int max) const {
  assert(min <= max);
  string value = getString(key);
  int x;
  if(!Global::tryStringToInt(value,x))
    throw new IOError("Could not parse '" + value + "' as int for key '" + key + "' in config file " + fileName);
  if(x < min || x > max)
    throw new IOError("Key '" + key + "' must be in the range " + Global::intToString(min) + " to " + Global::intToString(max) + " in config file " + fileName);
  return x;
}

int64_t ConfigParser::getInt64(const string& key) const {
  string value = getString(key);
  int64_t x;
  if(!Global::tryStringToInt64(value,x))
    throw new IOError("Could not parse '" + value + "' as int64_t for key '" + key + "' in config file " + fileName);
  return x;
}
int64_t ConfigParser::getInt64(const string& key, int64_t min, int64_t max) const {
  assert(min <= max);
  string value = getString(key);
  int64_t x;
  if(!Global::tryStringToInt64(value,x))
    throw new IOError("Could not parse '" + value + "' as int64_t for key '" + key + "' in config file " + fileName);
  if(x < min || x > max)
    throw new IOError("Key '" + key + "' must be in the range " + Global::int64ToString(min) + " to " + Global::int64ToString(max) + " in config file " + fileName);
  return x;
}


uint64_t ConfigParser::getUInt64(const string& key) const {
  string value = getString(key);
  uint64_t x;
  if(!Global::tryStringToUInt64(value,x))
    throw new IOError("Could not parse '" + value + "' as uint64_t for key '" + key + "' in config file " + fileName);
  return x;
}
uint64_t ConfigParser::getUInt64(const string& key, uint64_t min, uint64_t max) const {
  assert(min <= max);
  string value = getString(key);
  uint64_t x;
  if(!Global::tryStringToUInt64(value,x))
    throw new IOError("Could not parse '" + value + "' as int64_t for key '" + key + "' in config file " + fileName);
  if(x < min || x > max)
    throw new IOError("Key '" + key + "' must be in the range " + Global::uint64ToString(min) + " to " + Global::uint64ToString(max) + " in config file " + fileName);
  return x;
}

float ConfigParser::getFloat(const string& key) const {
  string value = getString(key);
  float x;
  if(!Global::tryStringToFloat(value,x))
    throw new IOError("Could not parse '" + value + "' as float for key '" + key + "' in config file " + fileName);
  return x;
}
float ConfigParser::getFloat(const string& key, float min, float max) const {
  assert(min <= max);
  string value = getString(key);
  float x;
  if(!Global::tryStringToFloat(value,x))
    throw new IOError("Could not parse '" + value + "' as float for key '" + key + "' in config file " + fileName);
  if(isnan(x))
    throw new IOError("Key '" + key + "' is nan in config file " + fileName);
  if(x < min || x > max)
    throw new IOError("Key '" + key + "' must be in the range " + Global::floatToString(min) + " to " + Global::floatToString(max) + " in config file " + fileName);
  return x;
}

double ConfigParser::getDouble(const string& key) const {
  string value = getString(key);
  double x;
  if(!Global::tryStringToDouble(value,x))
    throw new IOError("Could not parse '" + value + "' as double for key '" + key + "' in config file " + fileName);
  return x;
}
double ConfigParser::getDouble(const string& key, double min, double max) const {
  assert(min <= max);
  string value = getString(key);
  double x;
  if(!Global::tryStringToDouble(value,x))
    throw new IOError("Could not parse '" + value + "' as double for key '" + key + "' in config file " + fileName);
  if(isnan(x))
    throw new IOError("Key '" + key + "' is nan in config file " + fileName);
  if(x < min || x > max)
    throw new IOError("Key '" + key + "' must be in the range " + Global::doubleToString(min) + " to " + Global::doubleToString(max) + " in config file " + fileName);
  return x;
}
