#include "../core/config_parser.h"

#include "../core/fileutils.h"

#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

ConfigParser::ConfigParser(bool keysOverride, bool keysOverrideFromIncludes_)
  :initialized(false),fileName(),contents(),keyValues(),
    keysOverrideEnabled(keysOverride),keysOverrideFromIncludes(keysOverrideFromIncludes_),
    usedKeysMutex(),usedKeys()
{}

ConfigParser::ConfigParser(const string& fname, bool keysOverride, bool keysOverrideFromIncludes_)
  :initialized(false),fileName(),contents(),keyValues(),
    keysOverrideEnabled(keysOverride),keysOverrideFromIncludes(keysOverrideFromIncludes_),
    usedKeysMutex(),usedKeys()
{
  initialize(fname);
}

ConfigParser::ConfigParser(const char* fname, bool keysOverride, bool keysOverrideFromIncludes_)
  : ConfigParser(std::string(fname), keysOverride, keysOverrideFromIncludes_)
{}

ConfigParser::ConfigParser(istream& in, bool keysOverride, bool keysOverrideFromIncludes_)
  :initialized(false),fileName(),contents(),keyValues(),
    keysOverrideEnabled(keysOverride),keysOverrideFromIncludes(keysOverrideFromIncludes_),
    usedKeysMutex(),usedKeys()
{
  initialize(in);
}

ConfigParser::ConfigParser(const map<string, string>& kvs)
  :initialized(false),fileName(),contents(),keyValues(),
    keysOverrideEnabled(false),keysOverrideFromIncludes(true),
    usedKeysMutex(),usedKeys()
{
  initialize(kvs);
}

ConfigParser::ConfigParser(const ConfigParser& source) {
  if(!source.initialized)
    throw StringError("Can only copy a ConfigParser which has been initialized.");
  std::lock_guard<std::mutex> lock(source.usedKeysMutex);
  initialized = source.initialized;
  fileName = source.fileName;
  baseDirs = source.baseDirs;
  contents = source.contents;
  keyValues = source.keyValues;
  keysOverrideEnabled = source.keysOverrideEnabled;
  keysOverrideFromIncludes = source.keysOverrideFromIncludes;
  usedKeys = source.usedKeys;
}

void ConfigParser::initialize(const string& fname) {
  if(initialized)
    throw StringError("ConfigParser already initialized, cannot initialize again");
  ifstream in;
  FileUtils::open(in,fname);
  fileName = fname;
  string baseDir = extractBaseDir(fname);
  if (!baseDir.empty())
    baseDirs.push_back(baseDir);
  initializeInternal(in);
  initialized = true;
}

void ConfigParser::initialize(istream& in) {
  if(initialized)
    throw StringError("ConfigParser already initialized, cannot initialize again");
  initializeInternal(in);
  initialized = true;
}

void ConfigParser::initialize(const map<string, string>& kvs) {
  if(initialized)
    throw StringError("ConfigParser already initialized, cannot initialize again");
  keyValues = kvs;
  initialized = true;
}

void ConfigParser::initializeInternal(istream& in) {
  keyValues.clear();
  contents.clear();
  curFilename = fileName;
  readStreamContent(in);
}

void ConfigParser::processIncludedFile(const std::string &fname) {
  if(fname == fileName || find(includedFiles.begin(), includedFiles.end(), fname) != includedFiles.end()) {
    throw ConfigParsingError("Circular or multiple inclusion of the same file: '" + fname + "'" + lineAndFileInfo());
  }
  includedFiles.push_back(fname);
  curFilename = fname;

  string fpath;
  for(const auto &p: baseDirs) {
    fpath += p;
  }
  fpath += fname;

  string baseDir = extractBaseDir(fname);
  if(!baseDir.empty()) {
    if(baseDir[0] == '\\' || baseDir[0] == '/')
      throw ConfigParsingError("Absolute paths in the included files are not supported yet");
    baseDirs.push_back(baseDir);
  }

  ifstream in;
  FileUtils::open(in,fpath);
  readStreamContent(in);

  if(!baseDir.empty())
    baseDirs.pop_back();
}

void ConfigParser::readStreamContent(istream& in) {
  curLineNum = 0;
  string line;
  ostringstream contentStream;
  set<string> curFileKeys;
  while(getline(in,line)) {
    contentStream << line << "\n";
    curLineNum += 1;
    line = Global::trim(line);
    if(line.length() <= 0 || line[0] == '#')
      continue;

    size_t commentPos = line.find("#");
    if(commentPos != string::npos)
      line = line.substr(0, commentPos);

    if(line[0] == '@') {
      if(line.size() < 9) {
        throw ConfigParsingError("Unsupported @ directive" + lineAndFileInfo());
      }
      size_t pos0 =line.find_first_of(" \t\v\f=");
      if(pos0 == string::npos)
        throw ConfigParsingError("@ directive without value (key-val separator is not found)" + lineAndFileInfo());

      string key = Global::trim(line.substr(0,pos0));
      if(key != "@include")
        throw ConfigParsingError("Unsupported @ directive '" + key + "'" + lineAndFileInfo());

      string value = line.substr(pos0+1);
      size_t pos1 =value.find_first_not_of(" \t\v\f=");
      if(pos1 == string::npos)
        throw ConfigParsingError("@ directive without value (value after key-val separator is not found)" + lineAndFileInfo());

      value = Global::trim(value.substr(pos1));
      value = Global::trim(value, "'");  // remove single quotes for filename
      value = Global::trim(value, "\"");  // remove double quotes for filename

      int lineNum = curLineNum;
      processIncludedFile(value);
      curLineNum = lineNum;
      continue;
    }

    size_t pos = line.find("=");
    if(pos == string::npos)
      throw ConfigParsingError("Could not parse kv pair, line does not have a non-commented '='" + lineAndFileInfo());

    string key = Global::trim(line.substr(0,pos));
    string value = Global::trim(line.substr(pos+1));
    if(curFileKeys.find(key) != curFileKeys.end()) {
      if(!keysOverrideEnabled)
        throw ConfigParsingError("Key '" + key + "' + was specified multiple times in " +
                      curFilename + ", you probably didn't mean to do this, please delete one of them");
      else
        logMessages.push_back("Key '" + key + "' + was overriden by new value '" + value + "'" + lineAndFileInfo());
    }
    if(keyValues.find(key) != keyValues.end()) {
      if(!keysOverrideFromIncludes)
        throw ConfigParsingError("Key '" + key + "' + was specified multiple times in " +
                      curFilename + " or its included files, and key overriding is disabled");
      else
        logMessages.push_back("Key '" + key + "' + was overriden by new value '" + value + "'" + lineAndFileInfo());
    }
    keyValues[key] = value;
    curFileKeys.insert(key);
  }
  contents += contentStream.str();
}

string ConfigParser::lineAndFileInfo() const {
  return ", line " + Global::intToString(curLineNum) + " in '" + curFilename + "'";
}

string ConfigParser::extractBaseDir(const std::string &fname) {
  size_t slash = fname.find_last_of("/\\");
  if(slash != string::npos)
    return fname.substr(0, slash + 1);
  else
    return std::string();
}

ConfigParser::~ConfigParser()
{}

string ConfigParser::getFileName() const {
  return fileName;
}

string ConfigParser::getContents() const {
  return contents;
}

string ConfigParser::getAllKeyVals() const {
  ostringstream ost;
  for(auto it = keyValues.begin(); it != keyValues.end(); ++it) {
    ost << it->first + " = " + it->second << endl;
  }
  return ost.str();
}

void ConfigParser::unsetUsedKey(const string& key) {
  std::lock_guard<std::mutex> lock(usedKeysMutex);
  usedKeys.erase(key);
}

void ConfigParser::applyAlias(const string& mapThisKey, const string& toThisKey) {
  if(contains(mapThisKey) && contains(toThisKey))
    throw IOError("Cannot specify both " + mapThisKey + " and " + toThisKey + " in the same config");
  if(contains(mapThisKey)) {
    keyValues[toThisKey] = keyValues[mapThisKey];
    keyValues.erase(mapThisKey);
    std::lock_guard<std::mutex> lock(usedKeysMutex);
    if(usedKeys.find(mapThisKey) != usedKeys.end()) {
      usedKeys.insert(toThisKey);
      usedKeys.erase(mapThisKey);
    }
  }
}

void ConfigParser::overrideKey(const std::string& key, const std::string& value) {
  //Assume zero-length values mean to delete a key
  if(value.length() <= 0) {
    if(keyValues.find(key) != keyValues.end())
      keyValues.erase(key);
  }
  else
    keyValues[key] = value;
}

void ConfigParser::overrideKeys(const std::string& fname) {
  // it's a new config file, so baseDir is not relevant anymore
  baseDirs.clear();
  processIncludedFile(fname);
}

void ConfigParser::overrideKeys(const map<string, string>& newkvs) {
  for(auto iter = newkvs.begin(); iter != newkvs.end(); ++iter) {
    //Assume zero-length values mean to delete a key
    if(iter->second.length() <= 0) {
      if(keyValues.find(iter->first) != keyValues.end())
        keyValues.erase(iter->first);
    }
    else
      keyValues[iter->first] = iter->second;
  }
  fileName += " and/or command-line and query overrides";
}


void ConfigParser::overrideKeys(const map<string, string>& newkvs, const vector<pair<set<string>,set<string>>>& mutexKeySets) {
  for(size_t i = 0; i<mutexKeySets.size(); i++) {
    const set<string>& a = mutexKeySets[i].first;
    const set<string>& b = mutexKeySets[i].second;
    bool hasA = false;
    for(auto iter = a.begin(); iter != a.end(); ++iter) {
      if(newkvs.find(*iter) != newkvs.end()) {
        hasA = true;
        break;
      }
    }
    bool hasB = false;
    for(auto iter = b.begin(); iter != b.end(); ++iter) {
      if(newkvs.find(*iter) != newkvs.end()) {
        hasB = true;
        break;
      }
    }
    if(hasA) {
      for(auto iter = b.begin(); iter != b.end(); ++iter)
        keyValues.erase(*iter);
    }
    if(hasB) {
      for(auto iter = a.begin(); iter != a.end(); ++iter)
        keyValues.erase(*iter);
    }
  }

  overrideKeys(newkvs);
}

map<string,string> ConfigParser::parseCommaSeparated(const string& commaSeparatedValues) {
  map<string,string> keyValues;
  vector<string> pieces = Global::split(commaSeparatedValues,',');
  for(size_t i = 0; i<pieces.size(); i++) {
    string s = Global::trim(pieces[i]);
    if(s.length() <= 0)
      continue;
    size_t pos = s.find("=");
    if(pos == string::npos)
      throw ConfigParsingError("Could not parse kv pair, could not find '=' in:" + s);

    string key = Global::trim(s.substr(0,pos));
    string value = Global::trim(s.substr(pos+1));
    keyValues[key] = value;
  }
  return keyValues;
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
  vector<string> messages;
  if(unused.size() > 0) {
    messages.push_back("--------------");
    messages.push_back("WARNING: Config had unused keys! You may have a typo, an option you specified is being unused from " + fileName);
  }
  for(size_t i = 0; i<unused.size(); i++) {
    messages.push_back("WARNING: Unused key '" + unused[i] + "' in " + fileName);
  }
  if(unused.size() > 0) {
    messages.push_back("--------------");
  }

  if(logger != NULL) {
    for(size_t i = 0; i<messages.size(); i++)
      logger->write(messages[i]);
  }
  for(size_t i = 0; i<messages.size(); i++)
    out << messages[i] << endl;
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

bool ConfigParser::containsAny(const std::vector<std::string>& possibleKeys) const {
  for(const string& key : possibleKeys) {
    if(contains(key))
      return true;
  }
  return false;
}

std::string ConfigParser::firstFoundOrFail(const std::vector<std::string>& possibleKeys) const {
  for(const string& key : possibleKeys) {
    if(contains(key))
      return key;
  }
  string message = "Could not find key";
  for(const string& key : possibleKeys) {
    message += " '" + key + "'";
  }
  throw IOError(message + " in config file " + fileName);
}

std::string ConfigParser::firstFoundOrEmpty(const std::vector<std::string>& possibleKeys) const {
  for(const string& key : possibleKeys) {
    if(contains(key))
      return key;
  }
  return string();
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

vector<string> ConfigParser::getStringsNonEmptyTrim(const string& key) {
  vector<string> raw = Global::split(getString(key),',');
  vector<string> trimmed;
  for(size_t i = 0; i<raw.size(); i++) {
    string s = Global::trim(raw[i]);
    if(s.length() <= 0)
      continue;
    trimmed.push_back(s);
  }
  return trimmed;
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

enabled_t ConfigParser::getEnabled(const string& key) {
  string value = Global::trim(Global::toLower(getString(key)));
  enabled_t x;
  if(!enabled_t::tryParse(value,x))
    throw IOError("Could not parse '" + value + "' as bool or auto for key '" + key + "' in config file " + fileName);
  return x;
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
