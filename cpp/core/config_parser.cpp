#include "../core/config_parser.h"

#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

ConfigParser::ConfigParser(const string& fname)
  :fileName(fname),contents(),keyValues(),usedKeysMutex(),usedKeys()
{
  ifstream in(fileName);
  if(!in.is_open())
    throw IOError("Could not open config file: " + fileName);

  int lineNum = 0;
  string line;
  ostringstream contentStream;
  while(getline(in,line)) {
    contentStream << line << "\n";
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
  contents = contentStream.str();
}

ConfigParser::ConfigParser(const map<string, string>& kvs)
  :fileName(),contents(),keyValues(kvs),usedKeysMutex(),usedKeys()
{}

ConfigParser::~ConfigParser()
{}

string ConfigParser::getFileName() const {
  return fileName;
}

string ConfigParser::getContents() const {
  return contents;
}

void ConfigParser::markAllKeysUsedWithPrefix(const string& prefix) {
  std::lock_guard<std::mutex> lock(usedKeysMutex);
  for(auto iter = keyValues.begin(); iter != keyValues.end(); ++iter) {
    const string& key = iter->first;
    if(Global::isPrefix(key,prefix))
      usedKeys.insert(key);
  }
}

void ConfigParser::warnUnusedKeys(ostream& out, Logger* logger) const {
  vector<string> unused = unusedKeys();
  for(size_t i = 0; i<unused.size(); i++) {
    string msg = "WARNING: Unused key '" + unused[i] + "' in " + fileName;
    if(logger != NULL)
      logger->write(msg);
    out << msg << endl;
  }
}

vector<string> ConfigParser::unusedKeys() const {
  std::lock_guard<std::mutex> lock(usedKeysMutex);
  vector<string> unused;
  for(auto iter = keyValues.begin(); iter != keyValues.end(); ++iter) {
    const string& key = iter->first;
    if(usedKeys.find(key) == usedKeys.end())
      unused.push_back(key);
  }
  return unused;
}

bool ConfigParser::contains(const string& key) const {
  return keyValues.find(key) != keyValues.end();
}

string ConfigParser::getString(const string& key) {
  auto iter = keyValues.find(key);
  if(iter == keyValues.end())
    throw IOError("Could not find key '" + key + "' in config file " + fileName);

  {
    std::lock_guard<std::mutex> lock(usedKeysMutex);
    usedKeys.insert(key);
  }

  return iter->second;
}

string ConfigParser::getString(const string& key, const set<string>& possibles) {
  string value = getString(key);
  if(possibles.find(value) == possibles.end())
    throw IOError("Key '" + key + "' must be one of (" + Global::concat(possibles,"|") + ") in config file " + fileName);
  return value;
}

vector<string> ConfigParser::getStrings(const string& key) {
  return Global::split(getString(key),',');
}

vector<string> ConfigParser::getStrings(const string& key, const set<string>& possibles) {
  vector<string> values = getStrings(key);
  for(size_t i = 0; i<values.size(); i++) {
    const string& value = values[i];
    if(possibles.find(value) == possibles.end())
      throw IOError("Key '" + key + "' must be one of (" + Global::concat(possibles,"|") + ") in config file " + fileName);
  }
  return values;
}


bool ConfigParser::getBool(const string& key) {
  string value = getString(key);
  bool x;
  if(!Global::tryStringToBool(value,x))
    throw IOError("Could not parse '" + value + "' as bool for key '" + key + "' in config file " + fileName);
  return x;
}
vector<bool> ConfigParser::getBools(const string& key) {
  vector<string> values = getStrings(key);
  vector<bool> ret;
  for(size_t i = 0; i<values.size(); i++) {
    const string& value = values[i];
    bool x;
    if(!Global::tryStringToBool(value,x))
      throw IOError("Could not parse '" + value + "' as bool for key '" + key + "' in config file " + fileName);
    ret.push_back(x);
  }
  return ret;
}

int ConfigParser::getInt(const string& key) {
  string value = getString(key);
  int x;
  if(!Global::tryStringToInt(value,x))
    throw IOError("Could not parse '" + value + "' as int for key '" + key + "' in config file " + fileName);
  return x;
}
int ConfigParser::getInt(const string& key, int min, int max) {
  assert(min <= max);
  string value = getString(key);
  int x;
  if(!Global::tryStringToInt(value,x))
    throw IOError("Could not parse '" + value + "' as int for key '" + key + "' in config file " + fileName);
  if(x < min || x > max)
    throw IOError("Key '" + key + "' must be in the range " + Global::intToString(min) + " to " + Global::intToString(max) + " in config file " + fileName);
  return x;
}
vector<int> ConfigParser::getInts(const string& key) {
  vector<string> values = getStrings(key);
  vector<int> ret;
  for(size_t i = 0; i<values.size(); i++) {
    const string& value = values[i];
    int x;
    if(!Global::tryStringToInt(value,x))
      throw IOError("Could not parse '" + value + "' as int for key '" + key + "' in config file " + fileName);
    ret.push_back(x);
  }
  return ret;
}
vector<int> ConfigParser::getInts(const string& key, int min, int max) {
  vector<string> values = getStrings(key);
  vector<int> ret;
  for(size_t i = 0; i<values.size(); i++) {
    const string& value = values[i];
    int x;
    if(!Global::tryStringToInt(value,x))
      throw IOError("Could not parse '" + value + "' as int for key '" + key + "' in config file " + fileName);
    if(x < min || x > max)
      throw IOError("Key '" + key + "' must be in the range " + Global::intToString(min) + " to " + Global::intToString(max) + " in config file " + fileName);
    ret.push_back(x);
  }
  return ret;
}


int64_t ConfigParser::getInt64(const string& key) {
  string value = getString(key);
  int64_t x;
  if(!Global::tryStringToInt64(value,x))
    throw IOError("Could not parse '" + value + "' as int64_t for key '" + key + "' in config file " + fileName);
  return x;
}
int64_t ConfigParser::getInt64(const string& key, int64_t min, int64_t max) {
  assert(min <= max);
  string value = getString(key);
  int64_t x;
  if(!Global::tryStringToInt64(value,x))
    throw IOError("Could not parse '" + value + "' as int64_t for key '" + key + "' in config file " + fileName);
  if(x < min || x > max)
    throw IOError("Key '" + key + "' must be in the range " + Global::int64ToString(min) + " to " + Global::int64ToString(max) + " in config file " + fileName);
  return x;
}
vector<int64_t> ConfigParser::getInt64s(const string& key) {
  vector<string> values = getStrings(key);
  vector<int64_t> ret;
  for(size_t i = 0; i<values.size(); i++) {
    const string& value = values[i];
    int64_t x;
    if(!Global::tryStringToInt64(value,x))
      throw IOError("Could not parse '" + value + "' as int64_t for key '" + key + "' in config file " + fileName);
    ret.push_back(x);
  }
  return ret;
}
vector<int64_t> ConfigParser::getInt64s(const string& key, int64_t min, int64_t max) {
  vector<string> values = getStrings(key);
  vector<int64_t> ret;
  for(size_t i = 0; i<values.size(); i++) {
    const string& value = values[i];
    int64_t x;
    if(!Global::tryStringToInt64(value,x))
      throw IOError("Could not parse '" + value + "' as int64_t for key '" + key + "' in config file " + fileName);
    if(x < min || x > max)
      throw IOError("Key '" + key + "' must be in the range " + Global::int64ToString(min) + " to " + Global::int64ToString(max) + " in config file " + fileName);
    ret.push_back(x);
  }
  return ret;
}


uint64_t ConfigParser::getUInt64(const string& key) {
  string value = getString(key);
  uint64_t x;
  if(!Global::tryStringToUInt64(value,x))
    throw IOError("Could not parse '" + value + "' as uint64_t for key '" + key + "' in config file " + fileName);
  return x;
}
uint64_t ConfigParser::getUInt64(const string& key, uint64_t min, uint64_t max) {
  assert(min <= max);
  string value = getString(key);
  uint64_t x;
  if(!Global::tryStringToUInt64(value,x))
    throw IOError("Could not parse '" + value + "' as int64_t for key '" + key + "' in config file " + fileName);
  if(x < min || x > max)
    throw IOError("Key '" + key + "' must be in the range " + Global::uint64ToString(min) + " to " + Global::uint64ToString(max) + " in config file " + fileName);
  return x;
}
vector<uint64_t> ConfigParser::getUInt64s(const string& key) {
  vector<string> values = getStrings(key);
  vector<uint64_t> ret;
  for(size_t i = 0; i<values.size(); i++) {
    const string& value = values[i];
    uint64_t x;
    if(!Global::tryStringToUInt64(value,x))
      throw IOError("Could not parse '" + value + "' as uint64_t for key '" + key + "' in config file " + fileName);
    ret.push_back(x);
  }
  return ret;
}
vector<uint64_t> ConfigParser::getUInt64s(const string& key, uint64_t min, uint64_t max) {
  vector<string> values = getStrings(key);
  vector<uint64_t> ret;
  for(size_t i = 0; i<values.size(); i++) {
    const string& value = values[i];
    uint64_t x;
    if(!Global::tryStringToUInt64(value,x))
      throw IOError("Could not parse '" + value + "' as uint64_t for key '" + key + "' in config file " + fileName);
    if(x < min || x > max)
      throw IOError("Key '" + key + "' must be in the range " + Global::uint64ToString(min) + " to " + Global::uint64ToString(max) + " in config file " + fileName);
    ret.push_back(x);
  }
  return ret;
}


float ConfigParser::getFloat(const string& key) {
  string value = getString(key);
  float x;
  if(!Global::tryStringToFloat(value,x))
    throw IOError("Could not parse '" + value + "' as float for key '" + key + "' in config file " + fileName);
  return x;
}
float ConfigParser::getFloat(const string& key, float min, float max) {
  assert(min <= max);
  string value = getString(key);
  float x;
  if(!Global::tryStringToFloat(value,x))
    throw IOError("Could not parse '" + value + "' as float for key '" + key + "' in config file " + fileName);
  if(isnan(x))
    throw IOError("Key '" + key + "' is nan in config file " + fileName);
  if(x < min || x > max)
    throw IOError("Key '" + key + "' must be in the range " + Global::floatToString(min) + " to " + Global::floatToString(max) + " in config file " + fileName);
  return x;
}
vector<float> ConfigParser::getFloats(const string& key) {
  vector<string> values = getStrings(key);
  vector<float> ret;
  for(size_t i = 0; i<values.size(); i++) {
    const string& value = values[i];
    float x;
    if(!Global::tryStringToFloat(value,x))
      throw IOError("Could not parse '" + value + "' as float for key '" + key + "' in config file " + fileName);
    ret.push_back(x);
  }
  return ret;
}
vector<float> ConfigParser::getFloats(const string& key, float min, float max) {
  vector<string> values = getStrings(key);
  vector<float> ret;
  for(size_t i = 0; i<values.size(); i++) {
    const string& value = values[i];
    float x;
    if(!Global::tryStringToFloat(value,x))
      throw IOError("Could not parse '" + value + "' as float for key '" + key + "' in config file " + fileName);
    if(isnan(x))
      throw IOError("Key '" + key + "' is nan in config file " + fileName);
    if(x < min || x > max)
      throw IOError("Key '" + key + "' must be in the range " + Global::floatToString(min) + " to " + Global::floatToString(max) + " in config file " + fileName);
    ret.push_back(x);
  }
  return ret;
}


double ConfigParser::getDouble(const string& key) {
  string value = getString(key);
  double x;
  if(!Global::tryStringToDouble(value,x))
    throw IOError("Could not parse '" + value + "' as double for key '" + key + "' in config file " + fileName);
  return x;
}
double ConfigParser::getDouble(const string& key, double min, double max) {
  assert(min <= max);
  string value = getString(key);
  double x;
  if(!Global::tryStringToDouble(value,x))
    throw IOError("Could not parse '" + value + "' as double for key '" + key + "' in config file " + fileName);
  if(isnan(x))
    throw IOError("Key '" + key + "' is nan in config file " + fileName);
  if(x < min || x > max)
    throw IOError("Key '" + key + "' must be in the range " + Global::doubleToString(min) + " to " + Global::doubleToString(max) + " in config file " + fileName);
  return x;
}
vector<double> ConfigParser::getDoubles(const string& key) {
  vector<string> values = getStrings(key);
  vector<double> ret;
  for(size_t i = 0; i<values.size(); i++) {
    const string& value = values[i];
    double x;
    if(!Global::tryStringToDouble(value,x))
      throw IOError("Could not parse '" + value + "' as double for key '" + key + "' in config file " + fileName);
    ret.push_back(x);
  }
  return ret;
}
vector<double> ConfigParser::getDoubles(const string& key, double min, double max) {
  vector<string> values = getStrings(key);
  vector<double> ret;
  for(size_t i = 0; i<values.size(); i++) {
    const string& value = values[i];
    double x;
    if(!Global::tryStringToDouble(value,x))
      throw IOError("Could not parse '" + value + "' as double for key '" + key + "' in config file " + fileName);
    if(isnan(x))
      throw IOError("Key '" + key + "' is nan in config file " + fileName);
    if(x < min || x > max)
      throw IOError("Key '" + key + "' must be in the range " + Global::doubleToString(min) + " to " + Global::doubleToString(max) + " in config file " + fileName);
    ret.push_back(x);
  }
  return ret;
}
