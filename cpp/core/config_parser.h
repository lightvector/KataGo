#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include "../core/global.h"

/* Parses simple configs like:

#This is a comment
foo = true
bar = 64
baz = yay

*/

class ConfigParser {
 public:
  ConfigParser(const string& file);
  ~ConfigParser();

  ConfigParser(const ConfigParser& other) = delete;
  ConfigParser& operator=(const ConfigParser& other) = delete;
  ConfigParser(ConfigParser&& other) = delete;
  ConfigParser& operator=(ConfigParser&& other) = delete;

  vector<string> unusedKeys() const;

  bool contains(const string& key) const;

  string getString(const string& key);
  bool getBool(const string& key);
  int getInt(const string& key);
  int64_t getInt64(const string& key);
  uint64_t getUInt64(const string& key);
  float getFloat(const string& key);
  double getDouble(const string& key);

  string getString(const string& key, const set<string>& possibles);
  int getInt(const string& key, int min, int max);
  int64_t getInt64(const string& key, int64_t min, int64_t max);
  uint64_t getUInt64(const string& key, uint64_t min, uint64_t max);
  float getFloat(const string& key, float min, float max);
  double getDouble(const string& key, double min, double max);

  vector<string> getStrings(const string& key);
  vector<bool> getBools(const string& key);
  vector<int> getInts(const string& key);
  vector<int64_t> getInt64s(const string& key);
  vector<uint64_t> getUInt64s(const string& key);
  vector<float> getFloats(const string& key);
  vector<double> getDoubles(const string& key);

  vector<string> getStrings(const string& key, const set<string>& possibles);
  vector<int> getInts(const string& key, int min, int max);
  vector<int64_t> getInt64s(const string& key, int64_t min, int64_t max);
  vector<uint64_t> getUInt64s(const string& key, uint64_t min, uint64_t max);
  vector<float> getFloats(const string& key, float min, float max);
  vector<double> getDoubles(const string& key, double min, double max);

 private:
  string fileName;
  map<string,string> keyValues;
  set<string> usedKeys;
};



#endif
