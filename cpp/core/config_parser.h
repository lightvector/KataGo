#ifndef CORE_CONFIG_PARSER_H_
#define CORE_CONFIG_PARSER_H_

#include <mutex>

#include "../core/global.h"
#include "../core/logger.h"
#include "../core/commontypes.h"

/* Parses simple configs like:

#This is a comment
foo = true
bar = 64
baz = yay

*/

class ConfigParser {
 public:
  ConfigParser();
  ConfigParser(const std::string& file);
  ConfigParser(std::istream& in);
  ConfigParser(const std::map<std::string, std::string>& kvs);
  ConfigParser(const ConfigParser& source);
  ~ConfigParser();

  ConfigParser& operator=(const ConfigParser& other) = delete;
  ConfigParser(ConfigParser&& other) = delete;
  ConfigParser& operator=(ConfigParser&& other) = delete;

  void initialize(const std::string& file);
  void initialize(std::istream& in);
  void initialize(const std::map<std::string, std::string>& kvs);

  void overrideKey(const std::string& key, const std::string& value);
  void overrideKeys(const std::map<std::string, std::string>& newkvs);
  //mutexKeySets: For each pair of sets (A,B), if newkvs contains anything in A, erase every existing key that overlaps with B, and vice versa.
  void overrideKeys(const std::map<std::string, std::string>& newkvs, const std::vector<std::pair<std::set<std::string>,std::set<std::string>>>& mutexKeySets);
  static std::map<std::string,std::string> parseCommaSeparated(const std::string& commaSeparatedValues);

  void warnUnusedKeys(std::ostream& out, Logger* logger) const;
  void markAllKeysUsedWithPrefix(const std::string& prefix);
  void unsetUsedKey(const std::string& key);
  void applyAlias(const std::string& mapThisKey, const std::string& toThisKey);

  std::vector<std::string> unusedKeys() const;
  std::string getFileName() const;
  std::string getContents() const;

  bool contains(const std::string& key) const;
  bool containsAny(const std::vector<std::string>& possibleKeys) const;
  std::string firstFoundOrFail(const std::vector<std::string>& possibleKeys) const;
  std::string firstFoundOrEmpty(const std::vector<std::string>& possibleKeys) const;

  std::string getString(const std::string& key);
  bool getBool(const std::string& key);
  enabled_t getEnabled(const std::string& key);
  int getInt(const std::string& key);
  int64_t getInt64(const std::string& key);
  uint64_t getUInt64(const std::string& key);
  float getFloat(const std::string& key);
  double getDouble(const std::string& key);

  std::string getString(const std::string& key, const std::set<std::string>& possibles);
  int getInt(const std::string& key, int min, int max);
  int64_t getInt64(const std::string& key, int64_t min, int64_t max);
  uint64_t getUInt64(const std::string& key, uint64_t min, uint64_t max);
  float getFloat(const std::string& key, float min, float max);
  double getDouble(const std::string& key, double min, double max);

  std::vector<std::string> getStrings(const std::string& key);
  std::vector<std::string> getStringsNonEmptyTrim(const std::string& key);
  std::vector<bool> getBools(const std::string& key);
  std::vector<int> getInts(const std::string& key);
  std::vector<int64_t> getInt64s(const std::string& key);
  std::vector<uint64_t> getUInt64s(const std::string& key);
  std::vector<float> getFloats(const std::string& key);
  std::vector<double> getDoubles(const std::string& key);

  std::vector<std::string> getStrings(const std::string& key, const std::set<std::string>& possibles);
  std::vector<int> getInts(const std::string& key, int min, int max);
  std::vector<int64_t> getInt64s(const std::string& key, int64_t min, int64_t max);
  std::vector<uint64_t> getUInt64s(const std::string& key, uint64_t min, uint64_t max);
  std::vector<float> getFloats(const std::string& key, float min, float max);
  std::vector<double> getDoubles(const std::string& key, double min, double max);

 private:
  bool initialized;
  std::string fileName;
  std::string contents;
  std::map<std::string, std::string> keyValues;

  mutable std::mutex usedKeysMutex;
  std::set<std::string> usedKeys;

  void initializeInternal(std::istream& in);
};



#endif  // CORE_CONFIG_PARSER_H_
