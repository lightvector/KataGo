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

  bool contains(const string& key) const;

  string getString(const string& key) const;
  bool getBool(const string& key) const;
  int getInt(const string& key) const;
  int64_t getInt64(const string& key) const;
  uint64_t getUInt64(const string& key) const;
  float getFloat(const string& key) const;
  double getDouble(const string& key) const;

  string getString(const string& key, const set<string>& possibles) const;
  int getInt(const string& key, int min, int max) const;
  int64_t getInt64(const string& key, int64_t min, int64_t max) const;
  uint64_t getUInt64(const string& key, uint64_t min, uint64_t max) const;
  float getFloat(const string& key, float min, float max) const;
  double getDouble(const string& key, double min, double max) const;

 private:
  string fileName;
  map<string,string> keyValues;
};



#endif
