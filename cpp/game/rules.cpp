#include "../game/rules.h"

#include "../external/nlohmann_json/json.hpp"

#include <sstream>

using namespace std;
using json = nlohmann::json;

Rules::Rules() {
  //Defaults if not set - closest match to TT rules
  koRule = KO_POSITIONAL;
  scoringRule = SCORING_AREA;
  taxRule = TAX_NONE;
  multiStoneSuicideLegal = true;
  komi = 7.5f;
}

Rules::Rules(int kRule, int sRule, int tRule, bool suic, float km)
  :koRule(kRule),scoringRule(sRule),taxRule(tRule),multiStoneSuicideLegal(suic),komi(km)
{}

Rules::~Rules() {
}

bool Rules::operator==(const Rules& other) const {
  return
    koRule == other.koRule &&
    scoringRule == other.scoringRule &&
    taxRule == other.taxRule &&
    multiStoneSuicideLegal == other.multiStoneSuicideLegal &&
    komi == other.komi;
}

bool Rules::operator!=(const Rules& other) const {
  return
    koRule != other.koRule ||
    scoringRule != other.scoringRule ||
    taxRule != other.taxRule ||
    multiStoneSuicideLegal != other.multiStoneSuicideLegal ||
    komi != other.komi;
}

Rules Rules::getTrompTaylorish() {
  Rules rules;
  rules.koRule = KO_POSITIONAL;
  rules.scoringRule = SCORING_AREA;
  rules.taxRule = TAX_NONE;
  rules.multiStoneSuicideLegal = true;
  rules.komi = 7.5f;
  return rules;
}

Rules Rules::getSimpleTerritory() {
  Rules rules;
  rules.koRule = KO_SIMPLE;
  rules.scoringRule = SCORING_TERRITORY;
  rules.taxRule = TAX_SEKI;
  rules.multiStoneSuicideLegal = false;
  rules.komi = 7.5f;
  return rules;
}

bool Rules::komiIsIntOrHalfInt(float komi) {
  return std::isfinite(komi) && komi * 2 == (int)(komi * 2);
}

set<string> Rules::koRuleStrings() {
  return {"SIMPLE","POSITIONAL","SITUATIONAL","SPIGHT"};
}
set<string> Rules::scoringRuleStrings() {
  return {"AREA","TERRITORY"};
}
set<string> Rules::taxRuleStrings() {
  return {"NONE","SEKI","ALL"};
}

int Rules::parseKoRule(const string& s) {
  if(s == "SIMPLE") return Rules::KO_SIMPLE;
  else if(s == "POSITIONAL") return Rules::KO_POSITIONAL;
  else if(s == "SITUATIONAL") return Rules::KO_SITUATIONAL;
  else if(s == "SPIGHT") return Rules::KO_SPIGHT;
  else throw StringError("Rules::parseKoRule: Invalid ko rule: " + s);
}
int Rules::parseScoringRule(const string& s) {
  if(s == "AREA") return Rules::SCORING_AREA;
  else if(s == "TERRITORY") return Rules::SCORING_TERRITORY;
  else throw StringError("Rules::parseScoringRule: Invalid scoring rule: " + s);
}
int Rules::parseTaxRule(const string& s) {
  if(s == "NONE") return Rules::TAX_NONE;
  else if(s == "SEKI") return Rules::TAX_SEKI;
  else if(s == "ALL") return Rules::TAX_ALL;
  else throw StringError("Rules::parseTaxRule: Invalid tax rule: " + s);
}

string Rules::writeKoRule(int koRule) {
  if(koRule == Rules::KO_SIMPLE) return string("SIMPLE");
  if(koRule == Rules::KO_POSITIONAL) return string("POSITIONAL");
  if(koRule == Rules::KO_SITUATIONAL) return string("SITUATIONAL");
  if(koRule == Rules::KO_SPIGHT) return string("SPIGHT");
  return string("UNKNOWN");
}
string Rules::writeScoringRule(int scoringRule) {
  if(scoringRule == Rules::SCORING_AREA) return string("AREA");
  if(scoringRule == Rules::SCORING_TERRITORY) return string("TERRITORY");
  return string("UNKNOWN");
}
string Rules::writeTaxRule(int taxRule) {
  if(taxRule == Rules::TAX_NONE) return string("NONE");
  if(taxRule == Rules::TAX_SEKI) return string("SEKI");
  if(taxRule == Rules::TAX_ALL) return string("ALL");
  return string("UNKNOWN");
}

ostream& operator<<(ostream& out, const Rules& rules) {
  out << "ko" << Rules::writeKoRule(rules.koRule)
      << "score" << Rules::writeScoringRule(rules.scoringRule)
      << "tax" << Rules::writeTaxRule(rules.taxRule)
      << "sui" << rules.multiStoneSuicideLegal
      << "komi" << rules.komi;
  return out;
}

string Rules::toStringNoKomi() const {
  ostringstream out;
  out << "ko" << Rules::writeKoRule(koRule)
      << "score" << Rules::writeScoringRule(scoringRule)
      << "tax" << Rules::writeTaxRule(taxRule)
      << "sui" << multiStoneSuicideLegal;
  return out.str();
}

string Rules::toString() const {
  ostringstream out;
  out << (*this);
  return out.str();
}

string Rules::toJsonString() const {
  json ret;
  ret["ko"] = writeKoRule(koRule);
  ret["score"] = writeScoringRule(scoringRule);
  ret["tax"] = writeTaxRule(taxRule);
  ret["suicide"] = multiStoneSuicideLegal;
  ret["komi"] = komi;
  return ret.dump();
}

string Rules::toJsonStringNoKomi() const {
  json ret;
  ret["ko"] = writeKoRule(koRule);
  ret["score"] = writeScoringRule(scoringRule);
  ret["tax"] = writeTaxRule(taxRule);
  ret["suicide"] = multiStoneSuicideLegal;
  return ret.dump();
}

static bool tryParseRulesHelper(const string& sOrig, Rules& buf, bool allowKomi) {
  Rules rules;
  string lowercased = Global::toLower(sOrig);
  if(lowercased == "japanese" || lowercased == "korean") {
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.koRule = Rules::KO_SIMPLE;
    rules.taxRule = Rules::TAX_SEKI;
    rules.multiStoneSuicideLegal = false;
    rules.komi = 6.5;
  }
  else if(lowercased == "chinese") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_SIMPLE;
    rules.taxRule = Rules::TAX_NONE;
    rules.multiStoneSuicideLegal = false;
    rules.komi = 7.5;
  }
  else if(lowercased == "aga" || lowercased == "bga" || lowercased == "french") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_SITUATIONAL;
    rules.taxRule = Rules::TAX_NONE;
    rules.multiStoneSuicideLegal = false;
    rules.komi = 7.5;
  }
  else if(lowercased == "nz" || lowercased == "new zealand" || lowercased == "new-zealand" || lowercased == "new_zealand") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_SITUATIONAL;
    rules.taxRule = Rules::TAX_NONE;
    rules.multiStoneSuicideLegal = true;
    rules.komi = 7.5;
  }
  else if(lowercased == "tromp-taylor" || lowercased == "tromp_taylor" || lowercased == "tromp taylor" || lowercased == "tromptaylor") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.taxRule = Rules::TAX_NONE;
    rules.multiStoneSuicideLegal = true;
    rules.komi = 7.5;
  }
  else if(lowercased == "goe" || lowercased == "ing") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.taxRule = Rules::TAX_NONE;
    rules.multiStoneSuicideLegal = true;
    rules.komi = 7.5;
  }
  else if(sOrig.length() > 0 && sOrig[0] == '{') {
    //Default if not specified
    rules = Rules::getTrompTaylorish();
    bool taxSpecified = false;
    try {
      json input = json::parse(sOrig);
      string s;
      if(input.find("ko") != input.end()) {
        try { rules.koRule = Rules::parseKoRule(input["ko"].get<string>()); }
        catch(const StringError&) { return false; }
      }
      if(input.find("score") != input.end()) {
        try { rules.scoringRule = Rules::parseScoringRule(input["score"].get<string>()); }
        catch(const StringError&) { return false; }
      }
      if(input.find("tax") != input.end()) {
        try { rules.taxRule = Rules::parseTaxRule(input["tax"].get<string>()); taxSpecified = true; }
        catch(const StringError&) { return false; }
      }
      if(input.find("suicide") != input.end()) {
        try { rules.multiStoneSuicideLegal = input["suicide"].get<bool>(); }
        catch(const StringError&) { return false; }
      }
      if(input.find("sui") != input.end()) {
        try { rules.multiStoneSuicideLegal = input["sui"].get<bool>(); }
        catch(const StringError&) { return false; }
      }
      if(input.find("komi") != input.end()) {
        if(!allowKomi)
          return false;
        try { rules.komi = input["komi"].get<float>(); }
        catch(const StringError&) { return false; }
        if(rules.komi < Rules::MIN_USER_KOMI || rules.komi > Rules::MAX_USER_KOMI || !Rules::komiIsIntOrHalfInt(rules.komi))
          return false;
      }
    }
    catch(nlohmann::detail::exception& e) {
      return false;
    }

    if(!taxSpecified)
      rules.taxRule = (rules.scoringRule == Rules::SCORING_TERRITORY ? Rules::TAX_SEKI : Rules::TAX_NONE);
  }

  else {
    auto startsWithAndStrip = [](string& str, const string& prefix) {
      bool matches = str.length() >= prefix.length() && str.substr(0,prefix.length()) == prefix;
      if(matches)
        str = str.substr(prefix.length());
      str = Global::trim(str);
      return matches;
    };

    //Default if not specified
    rules = Rules::getTrompTaylorish();

    string s = sOrig;
    s = Global::trim(s);

    //But don't allow the empty string
    if(s.length() <= 0)
      return false;

    bool taxSpecified = false;
    while(true) {
      if(s.length() <= 0)
        break;

      if(startsWithAndStrip(s,"komi")) {
        if(!allowKomi)
          return false;
        int endIdx = 0;
        while(endIdx < s.length() && !Global::isAlpha(s[endIdx] && !Global::isWhitespace(s[endIdx])))
          endIdx++;
        float komi;
        bool suc = Global::tryStringToFloat(s.substr(0,endIdx),komi);
        if(!suc)
          return false;
        if(!std::isfinite(komi) || komi > 1e6 || komi < -1e6)
          return false;
        rules.komi = komi;
        s = s.substr(endIdx);
        s = Global::trim(s);
        continue;
      }
      if(startsWithAndStrip(s,"ko")) {
        if(startsWithAndStrip(s,"SIMPLE")) rules.koRule = Rules::KO_SIMPLE;
        else if(startsWithAndStrip(s,"POSITIONAL")) rules.koRule = Rules::KO_POSITIONAL;
        else if(startsWithAndStrip(s,"SITUATIONAL")) rules.koRule = Rules::KO_SITUATIONAL;
        else if(startsWithAndStrip(s,"SPIGHT")) rules.koRule = Rules::KO_SPIGHT;
        else return false;
        continue;
      }
      if(startsWithAndStrip(s,"score")) {
        if(startsWithAndStrip(s,"AREA")) rules.scoringRule = Rules::SCORING_AREA;
        else if(startsWithAndStrip(s,"TERRITORY")) rules.scoringRule = Rules::SCORING_TERRITORY;
        else return false;
        continue;
      }
      if(startsWithAndStrip(s,"tax")) {
        if(startsWithAndStrip(s,"NONE")) {rules.taxRule = Rules::TAX_NONE; taxSpecified = true;}
        else if(startsWithAndStrip(s,"SEKI")) {rules.taxRule = Rules::TAX_SEKI; taxSpecified = true;}
        else if(startsWithAndStrip(s,"ALL")) {rules.taxRule = Rules::TAX_ALL; taxSpecified = true;}
        else return false;
        continue;
      }
      if(startsWithAndStrip(s,"sui")) {
        if(startsWithAndStrip(s,"1")) rules.multiStoneSuicideLegal = true;
        else if(startsWithAndStrip(s,"0")) rules.multiStoneSuicideLegal = false;
        else return false;
        continue;
      }

      //Unknown rules format
      return false;
    }
    if(!taxSpecified)
      rules.taxRule = (rules.scoringRule == Rules::SCORING_TERRITORY ? Rules::TAX_SEKI : Rules::TAX_NONE);
  }

  buf = rules;
  return true;
}

bool Rules::tryParseRules(const string& sOrig, Rules& buf) {
  return tryParseRulesHelper(sOrig,buf,true);
}
bool Rules::tryParseRulesWithoutKomi(const string& sOrig, Rules& buf, float komi) {
  bool suc = tryParseRulesHelper(sOrig,buf,false);
  if(!suc)
    return false;
  buf.komi = komi;
  return true;
}


const Hash128 Rules::ZOBRIST_KO_RULE_HASH[4] = {
  Hash128(0x3cc7e0bf846820f6ULL, 0x1fb7fbde5fc6ba4eULL),  //Based on sha256 hash of Rules::KO_SIMPLE
  Hash128(0xcc18f5d47188554aULL, 0x3a63152c23e4128dULL),  //Based on sha256 hash of Rules::KO_POSITIONAL
  Hash128(0x3bc55e42b23b35bfULL, 0xc75fa1e615621dcdULL),  //Based on sha256 hash of Rules::KO_SITUATIONAL
  Hash128(0x5b2096e48241d21bULL, 0x23cc18d4e85cd67fULL),  //Based on sha256 hash of Rules::KO_SPIGHT
};

const Hash128 Rules::ZOBRIST_SCORING_RULE_HASH[2] = {
  //Based on sha256 hash of Rules::SCORING_AREA, but also mixing none tax rule hash, to preserve legacy hashes
  Hash128(0x8b3ed7598f901494ULL ^ 0x72eeccc72c82a5e7ULL, 0x1dfd47ac77bce5f8ULL ^ 0x0d1265e413623e2bULL),
  //Based on sha256 hash of Rules::SCORING_TERRITORY, but also mixing seki tax rule hash, to preserve legacy hashes
  Hash128(0x381345dc357ec982ULL ^ 0x125bfe48a41042d5ULL, 0x03ba55c026026b56ULL ^ 0x061866b5f2b98a79ULL),
};
const Hash128 Rules::ZOBRIST_TAX_RULE_HASH[3] = {
  Hash128(0x72eeccc72c82a5e7ULL, 0x0d1265e413623e2bULL),  //Based on sha256 hash of Rules::TAX_NONE
  Hash128(0x125bfe48a41042d5ULL, 0x061866b5f2b98a79ULL),  //Based on sha256 hash of Rules::TAX_SEKI
  Hash128(0xa384ece9d8ee713cULL, 0xfdc9f3b5d1f3732bULL),  //Based on sha256 hash of Rules::TAX_ALL
};

const Hash128 Rules::ZOBRIST_MULTI_STONE_SUICIDE_HASH =   //Based on sha256 hash of Rules::ZOBRIST_MULTI_STONE_SUICIDE_HASH
  Hash128(0xf9b475b3bbf35e37ULL, 0xefa19d8b1e5b3e5aULL);
