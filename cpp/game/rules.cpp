#include "../game/rules.h"

#include <sstream>

using namespace std;

Rules::Rules() {
  //Defaults if not set - closest match to TT rules
  koRule = KO_POSITIONAL;
  scoringRule = SCORING_AREA;
  multiStoneSuicideLegal = true;
  komi = 7.5f;
}

Rules::Rules(int kRule, int sRule, bool suic, float km)
  :koRule(kRule),scoringRule(sRule),multiStoneSuicideLegal(suic),komi(km)
{}

Rules::~Rules() {
}

bool Rules::operator==(const Rules& other) const {
  return
    koRule == other.koRule &&
    scoringRule == other.scoringRule &&
    multiStoneSuicideLegal == other.multiStoneSuicideLegal &&
    komi == other.komi;
}

bool Rules::operator!=(const Rules& other) const {
  return
    koRule != other.koRule ||
    scoringRule != other.scoringRule ||
    multiStoneSuicideLegal != other.multiStoneSuicideLegal ||
    komi != other.komi;
}

Rules Rules::getTrompTaylorish() {
  Rules rules;
  rules.koRule = KO_POSITIONAL;
  rules.scoringRule = SCORING_AREA;
  rules.multiStoneSuicideLegal = true;
  rules.komi = 7.5f;
  return rules;
}

Rules Rules::getSimpleTerritory() {
  Rules rules;
  rules.koRule = KO_SIMPLE;
  rules.scoringRule = SCORING_TERRITORY;
  rules.multiStoneSuicideLegal = false;
  rules.komi = 7.5f;
  return rules;
}

bool Rules::komiIsIntOrHalfInt(float komi) {
  return komi * 2 == (int)(komi * 2);
}

set<string> Rules::koRuleStrings() {
  return {"SIMPLE","POSITIONAL","SITUATIONAL","SPIGHT"};
}
set<string> Rules::scoringRuleStrings() {
  return {"AREA","TERRITORY"};
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

ostream& operator<<(ostream& out, const Rules& rules) {
  out << "ko" << Rules::writeKoRule(rules.koRule)
      << "score" << Rules::writeScoringRule(rules.scoringRule)
      << "sui" << rules.multiStoneSuicideLegal
      << "komi" << rules.komi;
  return out;
}

string Rules::toString() const {
  ostringstream out;
  out << (*this);
  return out.str();
}

bool Rules::tryParseRules(const string& ss, Rules& buf) {
  Rules rules;
  string s = Global::toLower(ss);
  if(s == "japanese" || s == "korean") {
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.koRule = Rules::KO_SIMPLE;
    rules.multiStoneSuicideLegal = false;
    rules.komi = 6.5;
  }
  else if(s == "chinese") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_SIMPLE;
    rules.multiStoneSuicideLegal = false;
    rules.komi = 7.5;
  }
  else if(s == "aga" || s == "bga" || s == "french") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_SITUATIONAL;
    rules.multiStoneSuicideLegal = false;
    rules.komi = 7.5;
  }
  else if(s == "nz" || s == "new zealand" || s == "new-zealand" || s == "new_zealand") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_SITUATIONAL;
    rules.multiStoneSuicideLegal = true;
    rules.komi = 7.5;
  }
  else if(s == "tromp-taylor" || s == "tromp_taylor" || s == "tromp taylor" || s == "tromptaylor") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.multiStoneSuicideLegal = true;
    rules.komi = 7.5;
  }
  else if(s == "goe" || s == "ing") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.multiStoneSuicideLegal = true;
    rules.komi = 7.5;
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
    rules = getTrompTaylorish();

    s = Global::trim(s);

    //But don't allow the empty string
    if(s.length() <= 0)
      return false;

    while(true) {
      if(s.length() <= 0)
        break;

      if(startsWithAndStrip(s,"ko")) {
        if(startsWithAndStrip(s,"simple")) rules.koRule = Rules::KO_SIMPLE;
        else if(startsWithAndStrip(s,"positional")) rules.koRule = Rules::KO_POSITIONAL;
        else if(startsWithAndStrip(s,"situational")) rules.koRule = Rules::KO_SITUATIONAL;
        else if(startsWithAndStrip(s,"spight")) rules.koRule = Rules::KO_SPIGHT;
        else return false;
        continue;
      }
      if(startsWithAndStrip(s,"score")) {
        if(startsWithAndStrip(s,"area")) rules.scoringRule = Rules::SCORING_AREA;
        else if(startsWithAndStrip(s,"territory")) rules.scoringRule = Rules::SCORING_TERRITORY;
        else return false;
        continue;
      }
      if(startsWithAndStrip(s,"sui")) {
        if(startsWithAndStrip(s,"1")) rules.multiStoneSuicideLegal = true;
        else if(startsWithAndStrip(s,"0")) rules.multiStoneSuicideLegal = false;
        else return false;
        continue;
      }
      if(startsWithAndStrip(s,"komi")) {
        int endIdx = 0;
        while(endIdx < s.length() && !Global::isAlpha(s[endIdx] && !Global::isWhitespace(s[endIdx])))
          endIdx++;
        float komi;
        bool suc = Global::tryStringToFloat(s.substr(endIdx),komi);
        if(!suc)
          return false;
        rules.komi = komi;
        s = s.substr(endIdx);
        s = Global::trim(s);
        continue;
      }

      //Unknown rules format
      return false;
    }
  }

  buf = rules;
  return true;
}

const Hash128 Rules::ZOBRIST_KO_RULE_HASH[4] = {
  Hash128(0x3cc7e0bf846820f6ULL, 0x1fb7fbde5fc6ba4eULL),  //Based on sha256 hash of Rules::KO_SIMPLE
  Hash128(0xcc18f5d47188554aULL, 0x3a63152c23e4128dULL),  //Based on sha256 hash of Rules::KO_POSITIONAL
  Hash128(0x3bc55e42b23b35bfULL, 0xc75fa1e615621dcdULL),  //Based on sha256 hash of Rules::KO_SITUATIONAL
  Hash128(0x5b2096e48241d21bULL, 0x23cc18d4e85cd67fULL),  //Based on sha256 hash of Rules::KO_SPIGHT
};

const Hash128 Rules::ZOBRIST_SCORING_RULE_HASH[2] = {
  Hash128(0x8b3ed7598f901494ULL, 0x1dfd47ac77bce5f8ULL),  //Based on sha256 hash of Rules::SCORING_AREA
  Hash128(0x381345dc357ec982ULL, 0x03ba55c026026b56ULL),  //Based on sha256 hash of Rules::SCORING_TERRITORY
};

const Hash128 Rules::ZOBRIST_MULTI_STONE_SUICIDE_HASH =  //Based on sha256 hash of Rules::ZOBRIST_MULTI_STONE_SUICIDE_HASH
  Hash128(0xf9b475b3bbf35e37ULL, 0xefa19d8b1e5b3e5aULL);
