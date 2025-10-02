#include "../game/rules.h"

#include "../external/nlohmann_json/json.hpp"

#include <sstream>

#include "../core/rand.h"
#include "board.h"

using namespace std;
using json = nlohmann::json;

const Rules Rules::DEFAULT_DOTS = Rules(true);
const Rules Rules::DEFAULT_GO = Rules(false);

Rules::Rules() : Rules(false) {}

Rules::Rules(const int startPos, const bool startPosIsRandom, const bool suicide, const bool dotsCaptureEmptyBases, const bool dotsFreeCapturedDots) :
  Rules(true, startPos, startPosIsRandom, 0, 0, 0, suicide, false, 0, false, 0.0f, dotsCaptureEmptyBases, dotsFreeCapturedDots) {}

Rules::Rules(
  int kRule,
  int sRule,
  int tRule,
  bool suic,
  bool button,
  int whbRule,
  bool pOk,
  float km
) : Rules(false, 0, false, kRule, sRule, tRule, suic, button, whbRule, pOk, km, false, false) {
}

Rules::Rules(const bool initIsDots) : Rules(
  initIsDots,
  initIsDots ? START_POS_CROSS : 0,
  false,
  initIsDots ? 0 : KO_POSITIONAL,
  initIsDots ? 0 : SCORING_AREA,
  initIsDots ? 0 : TAX_NONE,
  initIsDots,
  false,
  initIsDots ? 0 : WHB_ZERO,
  false,
  initIsDots ? 0.0f : 7.5f,
  false,
  initIsDots
  ) {
}

Rules::Rules(
  bool isDots,
  int startPosRule,
  bool startPosIsRandom,
  int kRule,
  int sRule,
  int tRule,
  bool suic,
  bool button,
  int whbRule,
  bool pOk,
  float km,
  bool dotsCaptureEmptyBases,
  bool dotsFreeCapturedDots
)
  : isDots(isDots),
    startPos(startPosRule),
    startPosIsRandom(startPosIsRandom),
    dotsCaptureEmptyBases(dotsCaptureEmptyBases),
    dotsFreeCapturedDots(dotsFreeCapturedDots),
    koRule(kRule),
    scoringRule(sRule),
    taxRule(tRule),
    multiStoneSuicideLegal(suic),
    hasButton(button),
    whiteHandicapBonusRule(whbRule),
    friendlyPassOk(pOk),
    komi(km)
{
  initializeIfNeeded();
}

Rules::~Rules() = default;

bool Rules::operator==(const Rules& other) const {
  return equals(other, false);
}

bool Rules::operator!=(const Rules& other) const {
  return !equals(other, false);
}

bool Rules::equalsIgnoringSgfDefinedProps(const Rules& other) const {
  return equals(other, true);
}

bool Rules::equals(const Rules& other, const bool ignoreSgfDefinedProps) const {
  return
    (ignoreSgfDefinedProps ? true : isDots == other.isDots) &&
    (ignoreSgfDefinedProps ? true : startPos == other.startPos) &&
    startPosIsRandom == other.startPosIsRandom &&
    koRule == other.koRule &&
    scoringRule == other.scoringRule &&
    taxRule == other.taxRule &&
    multiStoneSuicideLegal == other.multiStoneSuicideLegal &&
    hasButton == other.hasButton &&
    whiteHandicapBonusRule == other.whiteHandicapBonusRule &&
    friendlyPassOk == other.friendlyPassOk &&
    (ignoreSgfDefinedProps ? true : komi == other.komi) &&
    dotsCaptureEmptyBases == other.dotsCaptureEmptyBases &&
    dotsFreeCapturedDots == other.dotsFreeCapturedDots;
}

bool Rules::gameResultWillBeInteger() const {
  bool komiIsInteger = ((int)komi) == komi;
  return komiIsInteger != hasButton;
}

Rules Rules::getDefault(const bool isDots) {
  return isDots ? DEFAULT_DOTS : DEFAULT_GO;
}

Rules Rules::getDefaultOrTrompTaylorish(const bool isDots) {
  return isDots ? DEFAULT_DOTS : getTrompTaylorish();
}

Rules Rules::getTrompTaylorish() {
  Rules rules;
  rules.koRule = KO_POSITIONAL;
  rules.scoringRule = SCORING_AREA;
  rules.taxRule = TAX_NONE;
  rules.multiStoneSuicideLegal = true;
  rules.hasButton = false;
  rules.whiteHandicapBonusRule = WHB_ZERO;
  rules.friendlyPassOk = false;
  rules.komi = 7.5f;
  return rules;
}

Rules Rules::getSimpleTerritory() {
  Rules rules;
  rules.koRule = KO_SIMPLE;
  rules.scoringRule = SCORING_TERRITORY;
  rules.taxRule = TAX_SEKI;
  rules.multiStoneSuicideLegal = false;
  rules.hasButton = false;
  rules.whiteHandicapBonusRule = WHB_ZERO;
  rules.friendlyPassOk = false;
  rules.komi = 7.5f;
  return rules;
}

bool Rules::komiIsIntOrHalfInt(float komi) {
  return std::isfinite(komi) && komi * 2 == (int)(komi * 2);
}

set<string> Rules::startPosStrings() {
  initializeIfNeeded();
  return {
    startPosIdToName[START_POS_EMPTY],
    startPosIdToName[START_POS_CROSS],
    startPosIdToName[START_POS_CROSS_2],
    startPosIdToName[START_POS_CROSS_4],
    startPosIdToName[START_POS_CUSTOM]
  };
}

int Rules::getNumOfStartPosStones() const {
  switch (startPos) {
    case START_POS_EMPTY:
    case START_POS_CUSTOM:
      return 0;
    case START_POS_CROSS: return 4;
    case START_POS_CROSS_2: return 8;
    case START_POS_CROSS_4: return 16;
    default: throw std::range_error("Invalid start pos: " + std::to_string(startPos));
  }
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
set<string> Rules::whiteHandicapBonusRuleStrings() {
  return {"0","N","N-1"};
}

int Rules::parseStartPos(const string& s) {
  initializeIfNeeded();
  if (const auto it = startPosNameToId.find(s); it != startPosNameToId.end()) {
    return it->second;
  }

  throw IOError("Rules::parseStartPos: Invalid dots start pos rule: " + s);
}

int Rules::parseKoRule(const string& s) {
  if(s == "SIMPLE") return Rules::KO_SIMPLE;
  else if(s == "POSITIONAL") return Rules::KO_POSITIONAL;
  else if(s == "SITUATIONAL") return Rules::KO_SITUATIONAL;
  else if(s == "SPIGHT") return Rules::KO_SPIGHT;
  else throw IOError("Rules::parseKoRule: Invalid ko rule: " + s);
}

int Rules::parseScoringRule(const string& s) {
  if(s == "AREA") return Rules::SCORING_AREA;
  else if(s == "TERRITORY") return Rules::SCORING_TERRITORY;
  else throw IOError("Rules::parseScoringRule: Invalid scoring rule: " + s);
}

int Rules::parseTaxRule(const string& s) {
  if(s == "NONE") return Rules::TAX_NONE;
  else if(s == "SEKI") return Rules::TAX_SEKI;
  else if(s == "ALL") return Rules::TAX_ALL;
  else throw IOError("Rules::parseTaxRule: Invalid tax rule: " + s);
}

int Rules::parseWhiteHandicapBonusRule(const string& s) {
  if(s == "0") return Rules::WHB_ZERO;
  else if(s == "N") return Rules::WHB_N;
  else if(s == "N-1") return Rules::WHB_N_MINUS_ONE;
  else throw IOError("Rules::parseWhiteHandicapBonusRule: Invalid whiteHandicapBonus rule: " + s);
}

string Rules::writeStartPosRule(int startPosRule) {
  initializeIfNeeded();
  if (const auto it = startPosIdToName.find(startPosRule); it != startPosIdToName.end()) {
    return it->second;
  }

  return "UNKNOWN";
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
string Rules::writeWhiteHandicapBonusRule(int whiteHandicapBonusRule) {
  if(whiteHandicapBonusRule == Rules::WHB_ZERO) return string("0");
  if(whiteHandicapBonusRule == Rules::WHB_N) return string("N");
  if(whiteHandicapBonusRule == Rules::WHB_N_MINUS_ONE) return string("N-1");
  return string("UNKNOWN");
}

ostream& operator<<(ostream& out, const Rules& rules) {
  out << rules.toString();
  return out;
}

string Rules::toString(const bool includeSgfDefinedProperties) const {
  ostringstream out;
  if (!isDots) {
    out << "ko" << writeKoRule(koRule)
      << "score" << writeScoringRule(scoringRule)
      << "tax" << writeTaxRule(taxRule);
  } else {
    out << DOTS_CAPTURE_EMPTY_BASE_KEY << dotsCaptureEmptyBases;
  }
  if (includeSgfDefinedProperties && startPos != START_POS_EMPTY) {
    out << START_POS_KEY << writeStartPosRule(startPos);
  }
  if (startPosIsRandom) {
    out << START_POS_RANDOM_KEY << startPosIsRandom;
  }
  out << "sui" << multiStoneSuicideLegal;
  if (!isDots) {
    if (hasButton != DEFAULT_GO.hasButton)
      out << "button" << hasButton;
    if (whiteHandicapBonusRule != DEFAULT_GO.whiteHandicapBonusRule)
      out << "whb" << writeWhiteHandicapBonusRule(whiteHandicapBonusRule);
    if (friendlyPassOk != DEFAULT_GO.friendlyPassOk)
      out << "fpok" << friendlyPassOk;
  }
  if (includeSgfDefinedProperties) {
    out << "komi" << komi;
  }
  return out.str();
}

//omitDefaults: Takes up a lot of string space to include stuff, so omit some less common things if matches tromp-taylor rules
//which is the default for parsing and if not otherwise specified
json Rules::toJsonHelper(bool omitKomi, bool omitDefaults) const {
  json ret;
  if (isDots)
    ret[DOTS_KEY] = true;
  if (!omitDefaults || startPos != START_POS_EMPTY)
    ret[START_POS_KEY] = writeStartPosRule(startPos);
  ret["suicide"] = multiStoneSuicideLegal;
  if (!isDots) {
    ret["ko"] = writeKoRule(koRule);
    ret["scoring"] = writeScoringRule(scoringRule);
    ret["tax"] = writeTaxRule(taxRule);
    if(!omitDefaults || hasButton != DEFAULT_GO.hasButton)
      ret["hasButton"] = hasButton;
    if(!omitDefaults || whiteHandicapBonusRule != DEFAULT_GO.whiteHandicapBonusRule)
      ret["whiteHandicapBonus"] = writeWhiteHandicapBonusRule(whiteHandicapBonusRule);
    if(!omitDefaults || friendlyPassOk != DEFAULT_GO.friendlyPassOk)
      ret["friendlyPassOk"] = friendlyPassOk;
  } else {
    if (!omitDefaults || dotsCaptureEmptyBases != DEFAULT_DOTS.dotsCaptureEmptyBases)
      ret[DOTS_CAPTURE_EMPTY_BASE_KEY] = dotsCaptureEmptyBases;
    if (!omitDefaults || startPosIsRandom != DEFAULT_DOTS.startPosIsRandom)
      ret[START_POS_RANDOM_KEY] = startPosIsRandom;
  }
  if(!omitKomi)
    ret["komi"] = komi;
  return ret;
}

json Rules::toJson() const {
  return toJsonHelper(false,false);
}

json Rules::toJsonNoKomi() const {
  return toJsonHelper(true,false);
}

json Rules::toJsonNoKomiMaybeOmitStuff() const {
  return toJsonHelper(true,true);
}

string Rules::toJsonString() const {
  return toJsonHelper(false,false).dump();
}

string Rules::toJsonStringNoKomi() const {
  return toJsonHelper(true,false).dump();
}

string Rules::toJsonStringNoKomiMaybeOmitStuff() const {
  return toJsonHelper(true,true).dump();
}

Rules Rules::updateRules(const string& k, const string& v, Rules oldRules) {
  Rules rules = oldRules;
  string key = Global::trim(k);
  const string value = Global::trim(Global::toUpper(v));
  if(key == DOTS_KEY) rules.isDots = Global::stringToBool(value);
  else if(key == "ko") rules.koRule = Rules::parseKoRule(value);
  else if(key == "score") rules.scoringRule = Rules::parseScoringRule(value);
  else if(key == "scoring") rules.scoringRule = Rules::parseScoringRule(value);
  else if(key == "tax") rules.taxRule = Rules::parseTaxRule(value);
  else if(key == "suicide") rules.multiStoneSuicideLegal = Global::stringToBool(value);
  else if(key == "hasButton") rules.hasButton = Global::stringToBool(value);
  else if(key == "whiteHandicapBonus") rules.whiteHandicapBonusRule = Rules::parseWhiteHandicapBonusRule(value);
  else if(key == "friendlyPassOk") rules.friendlyPassOk = Global::stringToBool(value);
  else throw IOError("Unknown rules option: " + key);
  return rules;
}

static Rules parseRulesHelper(const string& sOrig, bool allowKomi, bool isDots) {
  auto rules = Rules(isDots);
  string lowercased = Global::trim(Global::toLower(sOrig));
  if(lowercased == DOTS_KEY) {
    rules = Rules::DEFAULT_DOTS;
  } else if(lowercased == "japanese" || lowercased == "korean") {
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.koRule = Rules::KO_SIMPLE;
    rules.taxRule = Rules::TAX_SEKI;
    rules.multiStoneSuicideLegal = false;
    rules.hasButton = false;
    rules.whiteHandicapBonusRule = Rules::WHB_ZERO;
    rules.friendlyPassOk = false;
    rules.komi = 6.5;
  }
  else if(lowercased == "chinese") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_SIMPLE;
    rules.taxRule = Rules::TAX_NONE;
    rules.multiStoneSuicideLegal = false;
    rules.hasButton = false;
    rules.whiteHandicapBonusRule = Rules::WHB_N;
    rules.friendlyPassOk = true;
    rules.komi = 7.5;
  }
  else if(
    lowercased == "chineseogs" || lowercased == "chinese_ogs" || lowercased == "chinese-ogs" ||
    lowercased == "chinesekgs" || lowercased == "chinese_kgs" || lowercased == "chinese-kgs"
  ) {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.taxRule = Rules::TAX_NONE;
    rules.multiStoneSuicideLegal = false;
    rules.hasButton = false;
    rules.whiteHandicapBonusRule = Rules::WHB_N;
    rules.friendlyPassOk = true;
    rules.komi = 7.5;
  }
  else if(
    lowercased == "ancientarea" || lowercased == "ancient-area" || lowercased == "ancient_area" || lowercased == "ancient area" ||
    lowercased == "stonescoring" || lowercased == "stone-scoring" || lowercased == "stone_scoring" || lowercased == "stone scoring"
  ) {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_SIMPLE;
    rules.taxRule = Rules::TAX_ALL;
    rules.multiStoneSuicideLegal = false;
    rules.hasButton = false;
    rules.whiteHandicapBonusRule = Rules::WHB_ZERO;
    rules.friendlyPassOk = true;
    rules.komi = 7.5;
  }
  else if(lowercased == "ancientterritory" || lowercased == "ancient-territory" || lowercased == "ancient_territory" || lowercased == "ancient territory") {
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.koRule = Rules::KO_SIMPLE;
    rules.taxRule = Rules::TAX_ALL;
    rules.multiStoneSuicideLegal = false;
    rules.hasButton = false;
    rules.whiteHandicapBonusRule = Rules::WHB_ZERO;
    rules.friendlyPassOk = false;
    rules.komi = 6.5;
  }
  else if(lowercased == "agabutton" || lowercased == "aga-button" || lowercased == "aga_button" || lowercased == "aga button") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_SITUATIONAL;
    rules.taxRule = Rules::TAX_NONE;
    rules.multiStoneSuicideLegal = false;
    rules.hasButton = true;
    rules.whiteHandicapBonusRule = Rules::WHB_N_MINUS_ONE;
    rules.friendlyPassOk = true;
    rules.komi = 7.0;
  }
  else if(lowercased == "aga" || lowercased == "bga" || lowercased == "french") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_SITUATIONAL;
    rules.taxRule = Rules::TAX_NONE;
    rules.multiStoneSuicideLegal = false;
    rules.hasButton = false;
    rules.whiteHandicapBonusRule = Rules::WHB_N_MINUS_ONE;
    rules.friendlyPassOk = true;
    rules.komi = 7.5;
  }
  else if(lowercased == "nz" || lowercased == "newzealand" || lowercased == "new zealand" || lowercased == "new-zealand" || lowercased == "new_zealand") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_SITUATIONAL;
    rules.taxRule = Rules::TAX_NONE;
    rules.multiStoneSuicideLegal = true;
    rules.hasButton = false;
    rules.whiteHandicapBonusRule = Rules::WHB_ZERO;
    rules.friendlyPassOk = true;
    rules.komi = 7.5;
  }
  else if(lowercased == "tromp-taylor" || lowercased == "tromp_taylor" || lowercased == "tromp taylor" || lowercased == "tromptaylor") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.taxRule = Rules::TAX_NONE;
    rules.multiStoneSuicideLegal = true;
    rules.hasButton = false;
    rules.whiteHandicapBonusRule = Rules::WHB_ZERO;
    rules.friendlyPassOk = false;
    rules.komi = 7.5;
  }
  else if(lowercased == "goe" || lowercased == "ing") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.taxRule = Rules::TAX_NONE;
    rules.multiStoneSuicideLegal = true;
    rules.hasButton = false;
    rules.whiteHandicapBonusRule = Rules::WHB_ZERO;
    rules.friendlyPassOk = true;
    rules.komi = 7.5;
  }
  else if(sOrig.length() > 0 && sOrig[0] == '{') {
    //Default if not specified
    rules = Rules::getDefaultOrTrompTaylorish(isDots);
    bool komiSpecified = false;
    bool taxSpecified = false;
    try {
      json input = json::parse(sOrig);
      string s;
      for(json::iterator iter = input.begin(); iter != input.end(); ++iter) {
        string key = iter.key();
        if (key == START_POS_KEY)
          rules.startPos = Rules::parseStartPos(iter.value().get<string>());
        else if (key == START_POS_RANDOM_KEY)
          rules.startPosIsRandom = iter.value().get<bool>();
        else if (key == DOTS_CAPTURE_EMPTY_BASE_KEY)
          rules.dotsCaptureEmptyBases = iter.value().get<bool>();
        else if(key == "ko")
          rules.koRule = Rules::parseKoRule(iter.value().get<string>());
        else if(key == "score" || key == "scoring")
          rules.scoringRule = Rules::parseScoringRule(iter.value().get<string>());
        else if(key == "tax") {
          rules.taxRule = Rules::parseTaxRule(iter.value().get<string>()); taxSpecified = true;
        }
        else if(key == "suicide")
          rules.multiStoneSuicideLegal = iter.value().get<bool>();
        else if(key == "hasButton")
          rules.hasButton = iter.value().get<bool>();
        else if(key == "whiteHandicapBonus")
          rules.whiteHandicapBonusRule = Rules::parseWhiteHandicapBonusRule(iter.value().get<string>());
        else if(key == "friendlyPassOk")
          rules.friendlyPassOk = iter.value().get<bool>();
        else if(key == "komi") {
          if(!allowKomi)
            throw IOError("Unknown rules option: " + key);
          rules.komi = iter.value().get<float>();
          komiSpecified = true;
          if(rules.komi < Rules::MIN_USER_KOMI || rules.komi > Rules::MAX_USER_KOMI || !Rules::komiIsIntOrHalfInt(rules.komi))
            throw IOError("Komi value is not a half-integer or is too extreme");
        }
        else
          throw IOError("Unknown rules option: " + key);
      }
    }
    catch(nlohmann::detail::exception&) {
      throw IOError("Could not parse rules: " + sOrig);
    }
    if(!taxSpecified)
      rules.taxRule = (rules.scoringRule == Rules::SCORING_TERRITORY ? Rules::TAX_SEKI : Rules::TAX_NONE);
    if(!komiSpecified) {
      //Drop default komi to 6.5 for territory rules, and to 7.0 for button
      if(rules.scoringRule == Rules::SCORING_TERRITORY)
        rules.komi = 6.5f;
      else if(rules.hasButton)
        rules.komi = 7.0f;
    }
  }

  //This is more of a legacy internal format, not recommended for users to provide
  else {
    auto startsWithAndStrip = [](string& str, const string& prefix) {
      bool matches = str.length() >= prefix.length() && str.substr(0,prefix.length()) == prefix;
      if(matches)
        str = str.substr(prefix.length());
      str = Global::trim(str);
      return matches;
    };

    //Default if not specified
    rules = Rules::getDefaultOrTrompTaylorish(isDots);

    string s = sOrig;
    s = Global::trim(s);

    //But don't allow the empty string
    if(s.length() <= 0)
      throw IOError("Could not parse rules: " + sOrig);

    bool komiSpecified = false;
    bool taxSpecified = false;
    while(true) {
      if(s.length() <= 0)
        break;

      if(startsWithAndStrip(s,"komi")) {
        if(!allowKomi)
          throw IOError("Could not parse rules: " + sOrig);
        int endIdx = 0;
        while(endIdx < s.length() && !Global::isAlpha(s[endIdx]) && !Global::isWhitespace(s[endIdx]))
          endIdx++;
        float komi;
        bool suc = Global::tryStringToFloat(s.substr(0,endIdx),komi);
        if(!suc)
          throw IOError("Could not parse rules: " + sOrig);
        if(!std::isfinite(komi) || komi > 1e5 || komi < -1e5)
          throw IOError("Could not parse rules: " + sOrig);
        rules.komi = komi;
        komiSpecified = true;
        s = s.substr(endIdx);
        s = Global::trim(s);
        continue;
      }
      if(startsWithAndStrip(s,"ko")) {
        if(startsWithAndStrip(s,"SIMPLE")) rules.koRule = Rules::KO_SIMPLE;
        else if(startsWithAndStrip(s,"POSITIONAL")) rules.koRule = Rules::KO_POSITIONAL;
        else if(startsWithAndStrip(s,"SITUATIONAL")) rules.koRule = Rules::KO_SITUATIONAL;
        else if(startsWithAndStrip(s,"SPIGHT")) rules.koRule = Rules::KO_SPIGHT;
        else throw IOError("Could not parse rules: " + sOrig);
        continue;
      }
      if(startsWithAndStrip(s,"scoring")) {
        if(startsWithAndStrip(s,"AREA")) rules.scoringRule = Rules::SCORING_AREA;
        else if(startsWithAndStrip(s,"TERRITORY")) rules.scoringRule = Rules::SCORING_TERRITORY;
        else throw IOError("Could not parse rules: " + sOrig);
        continue;
      }
      if(startsWithAndStrip(s,"score")) {
        if(startsWithAndStrip(s,"AREA")) rules.scoringRule = Rules::SCORING_AREA;
        else if(startsWithAndStrip(s,"TERRITORY")) rules.scoringRule = Rules::SCORING_TERRITORY;
        else throw IOError("Could not parse rules: " + sOrig);
        continue;
      }
      if(startsWithAndStrip(s,"tax")) {
        if(startsWithAndStrip(s,"NONE")) {rules.taxRule = Rules::TAX_NONE; taxSpecified = true;}
        else if(startsWithAndStrip(s,"SEKI")) {rules.taxRule = Rules::TAX_SEKI; taxSpecified = true;}
        else if(startsWithAndStrip(s,"ALL")) {rules.taxRule = Rules::TAX_ALL; taxSpecified = true;}
        else throw IOError("Could not parse rules: " + sOrig);
        continue;
      }
      if(startsWithAndStrip(s,"sui")) {
        if(startsWithAndStrip(s,"1")) rules.multiStoneSuicideLegal = true;
        else if(startsWithAndStrip(s,"0")) rules.multiStoneSuicideLegal = false;
        else throw IOError("Could not parse rules: " + sOrig);
        continue;
      }
      if(startsWithAndStrip(s,"button")) {
        if(startsWithAndStrip(s,"1")) rules.hasButton = true;
        else if(startsWithAndStrip(s,"0")) rules.hasButton = false;
        else throw IOError("Could not parse rules: " + sOrig);
        continue;
      }
      if(startsWithAndStrip(s,"whb")) {
        if(startsWithAndStrip(s,"0")) {rules.whiteHandicapBonusRule = Rules::WHB_ZERO;}
        else if(startsWithAndStrip(s,"N-1")) {rules.whiteHandicapBonusRule = Rules::WHB_N_MINUS_ONE;}
        else if(startsWithAndStrip(s,"N")) {rules.whiteHandicapBonusRule = Rules::WHB_N;}
        else throw IOError("Could not parse rules: " + sOrig);
        continue;
      }
      if(startsWithAndStrip(s,"fpok")) {
        if(startsWithAndStrip(s,"1")) rules.friendlyPassOk = true;
        else if(startsWithAndStrip(s,"0")) rules.friendlyPassOk = false;
        else throw IOError("Could not parse rules: " + sOrig);
        continue;
      }
      if(startsWithAndStrip(s,DOTS_CAPTURE_EMPTY_BASE_KEY)) {
        if(startsWithAndStrip(s,"1")) rules.dotsCaptureEmptyBases = true;
        else if(startsWithAndStrip(s,"0")) rules.dotsCaptureEmptyBases = false;
        else throw IOError("Could not parse rules: " + sOrig);
        continue;
      }
      if(startsWithAndStrip(s,START_POS_RANDOM_KEY)) {
        if(startsWithAndStrip(s,"1")) rules.startPosIsRandom = true;
        else if(startsWithAndStrip(s,"0")) rules.startPosIsRandom = false;
        else throw IOError("Could not parse rules: " + sOrig);
        continue;
      }

      //Unknown rules format
      else throw IOError("Could not parse rules: " + sOrig);
    }
    if(!taxSpecified)
      rules.taxRule = (rules.scoringRule == Rules::SCORING_TERRITORY ? Rules::TAX_SEKI : Rules::TAX_NONE);
    if(!komiSpecified) {
      //Drop default komi to 6.5 for territory rules, and to 7.0 for button
      if(rules.scoringRule == Rules::SCORING_TERRITORY)
        rules.komi = 6.5f;
      else if(rules.hasButton)
        rules.komi = 7.0f;
    }
  }

  return rules;
}

Rules Rules::parseRules(const string& sOrig, const bool isDots) {
  return parseRulesHelper(sOrig,true,isDots);
}

Rules Rules::parseRulesWithoutKomi(const string& sOrig, const float komi, const bool isDots) {
  Rules rules = parseRulesHelper(sOrig,false,isDots);
  rules.komi = komi;
  return rules;
}

bool Rules::tryParseRules(const string& sOrig, Rules& buf, bool isDots) {
  Rules rules;
  try { rules = parseRulesHelper(sOrig,true,isDots); }
  catch(const StringError&) { return false; }
  buf = rules;
  return true;
}

bool Rules::tryParseRulesWithoutKomi(const string& sOrig, Rules& buf, float komi, bool isDots) {
  Rules rules;
  try { rules = parseRulesHelper(sOrig,false,isDots); }
  catch(const StringError&) { return false; }
  rules.komi = komi;
  buf = rules;
  return true;
}

string Rules::toStringNoSgfDefinedPropertiesMaybeNice() const {
  if(equalsIgnoringSgfDefinedProps(parseRulesHelper(DOTS_KEY, false, isDots)))
    return "Dots";
  if(equalsIgnoringSgfDefinedProps(parseRulesHelper("TrompTaylor",false, isDots)))
    return "TrompTaylor";
  if(equalsIgnoringSgfDefinedProps(parseRulesHelper("Japanese",false, isDots)))
    return "Japanese";
  if(equalsIgnoringSgfDefinedProps(parseRulesHelper("Chinese",false, isDots)))
    return "Chinese";
  if(equalsIgnoringSgfDefinedProps(parseRulesHelper("Chinese-OGS",false, isDots)))
    return "Chinese-OGS";
  if(equalsIgnoringSgfDefinedProps(parseRulesHelper("AGA",false, isDots)))
    return "AGA";
  if(equalsIgnoringSgfDefinedProps(parseRulesHelper("StoneScoring",false, isDots)))
    return "StoneScoring";
  if(equalsIgnoringSgfDefinedProps(parseRulesHelper("NewZealand",false, isDots)))
    return "NewZealand";
  return toString(false);
}

double nextRandomOffset(Rand& rand) {
  return rand.nextDouble(4, 7);
}

int nextRandomOffsetX(Rand& rand, int x_size) {
  return static_cast<int>(round(nextRandomOffset(rand) / 39.0 * x_size));
}

int nextRandomOffsetY(Rand& rand, int y_size) {
  return static_cast<int>(round(nextRandomOffset(rand) / 32.0 * y_size));
}

std::vector<Move> Rules::generateStartPos(const int startPos, Rand* rand, const int x_size, const int y_size) {
  std::vector<Move> moves;
  switch (startPos) {
    case START_POS_EMPTY:
      break;
    case START_POS_CROSS:
      if (x_size >= 2 && y_size >= 2) {
        // Obey notago implementation for odd height
        addCross((x_size + 1) / 2 - 1, y_size / 2 - 1, x_size, false, moves);
      }
      break;
    case START_POS_CROSS_2:
      if (x_size >= 4 && y_size >= 2) {
        const int middleX = (x_size + 1) / 2;
        const int middleY = y_size / 2 - 1; // Obey notago implementation for odd height
        addCross(middleX - 2, middleY, x_size, false, moves);
        addCross(middleX, middleY, x_size, true, moves);
      }
      break;
    case START_POS_CROSS_4:
      if (x_size >= 4 && y_size >= 4) {
        int offsetX1;
        int offsetY1;
        int offsetX2;
        int offsetY2;
        int offsetX3;
        int offsetY3;
        int offsetX4;
        int offsetY4;

        if (rand == nullptr) {
          if (x_size == 39 && y_size == 32) {
            offsetX1 = 11;
            offsetY1 = 10;
          } else {
            offsetX1 = (x_size - 3) / 3;
            offsetY1 = (y_size - 3) / 3;
          }
          // Consider the index and size of the cross
          const int sideOffsetX = x_size - 1 - (offsetX1 + 1);
          const int sideOffsetY = y_size - 1 - (offsetY1 + 1);

          offsetX2 = sideOffsetX;
          offsetY2 = offsetY1;
          offsetX3 = sideOffsetX;
          offsetY3 = sideOffsetY;
          offsetX4 = offsetX1;
          offsetY4 = sideOffsetY;
        } else {
          const int middleX = x_size / 2 - 1;
          const int middleY = y_size / 2 - 1;

          offsetX1 = middleX - nextRandomOffsetX(*rand, x_size);
          offsetY1 = middleY - nextRandomOffsetY(*rand, y_size);
          offsetX2 = middleX + nextRandomOffsetX(*rand, x_size);
          offsetY2 = middleY - nextRandomOffsetY(*rand, y_size);
          offsetX3 = middleX + nextRandomOffsetX(*rand, x_size);
          offsetY3 = middleY + nextRandomOffsetY(*rand, y_size);
          offsetX4 = middleX - nextRandomOffsetX(*rand, x_size);
          offsetY4 = middleY + nextRandomOffsetY(*rand, y_size);
        }

        addCross(offsetX1, offsetY1, x_size, false, moves);
        addCross(offsetX2, offsetY2, x_size, false, moves);
        addCross(offsetX3, offsetY3, x_size, false, moves);
        addCross(offsetX4, offsetY4, x_size, false, moves);
      }
      break;
    default:
      throw std::range_error("Unsupported " + START_POS_KEY + ": " + to_string(startPos));
  }

  return moves;
}

void Rules::addCross(const int x, const int y, const int x_size, const bool rotate90, std::vector<Move>& moves) {
  Player pla;
  Player opp;

  // The end move should always be P_WHITE
  if (!rotate90) {
    pla = P_WHITE;
    opp = P_BLACK;
  } else {
    pla = P_BLACK;
    opp = P_WHITE;
  }

  const auto tailMove = Move(Location::getLoc(x, y + 1, x_size), pla);

  if (!rotate90) {
    moves.push_back(tailMove);
  }

  moves.emplace_back(Location::getLoc(x + 1, y + 1, x_size), opp);
  moves.emplace_back(Location::getLoc(x + 1, y, x_size), pla);
  moves.emplace_back(Location::getLoc(x, y, x_size), opp);

  if (rotate90) {
    moves.push_back(tailMove);
  }
}

int Rules::tryRecognizeStartPos(
  const vector<Move>& placementMoves,
  const int x_size,
  const int y_size,
  bool& randomized,
  vector<Move>* remainingMoves) {
  randomized = false; // Empty or unknown start pos is static by default
  if (remainingMoves != nullptr) {
    *remainingMoves = placementMoves;
  }
  int result = START_POS_EMPTY;

  if(placementMoves.empty()) return result;

  // If all placement moves are black, then it's a handicap game and the start pos is empty
  for (const auto placementMove : placementMoves) {
    if (placementMove.pla != C_BLACK) {
      result = START_POS_CUSTOM;
      break;
    }
  }

  if (result == START_POS_EMPTY) return result;

  const int stride = x_size + 1;
  auto placement = vector(stride * (y_size + 2), C_EMPTY);

  for (const auto move : placementMoves) {
    placement[move.loc] = move.pla;
  }

  auto recognizedCrossesMoves = vector<Move>();

  for (const auto move : placementMoves) {
    const int x = Location::getX(move.loc, x_size);
    const int y = Location::getY(move.loc, x_size);

    const int xy = Location::getLoc(x, y, x_size);
    const Player pla = placement[xy];
    if (pla == C_EMPTY) continue;

    const Player opp = getOpp(pla);
    if (opp == C_EMPTY) continue;

    if (x + 1 > x_size) continue;
    const int xp1y = Location::getLoc(x + 1, y, x_size);
    if (placement[xp1y] != opp) continue;

    if (y + 1 > y_size) continue;
    const int xp1yp1 = Location::getLoc(x + 1, y + 1, x_size);
    if (placement[xp1yp1] != pla) continue;

    const int xyp1 = Location::getLoc(x, y + 1, x_size);
    if (placement[xyp1] != opp) continue;

    recognizedCrossesMoves.emplace_back(xy, pla);
    recognizedCrossesMoves.emplace_back(xp1y, opp);
    recognizedCrossesMoves.emplace_back(xp1yp1, pla);
    recognizedCrossesMoves.emplace_back(xyp1, opp);

    // Clear the placement because the recognized cross is already stored
    placement[xy] = C_EMPTY;
    placement[xp1y] = C_EMPTY;
    placement[xp1yp1] = C_EMPTY;
    placement[xyp1] = C_EMPTY;
  }

  // Sort locs because start pos is invariant to moves order
  auto sortByLoc = [&](vector<Move>& moves) {
    std::sort(moves.begin(), moves.end(), [](const Move& move1, const Move& move2) { return move1.loc < move2.loc; });
  };

  sortByLoc(recognizedCrossesMoves);

  // Try to match strictly and set up randomized if failed.
  auto detectRandomization = [&](const int expectedStartPos) -> void {
    auto staticStartPosMoves = generateStartPos(expectedStartPos, nullptr, x_size, y_size);

    assert(remainingMoves != nullptr || placementMoves.size() == recognizedCrossesMoves.size());
    assert(staticStartPosMoves.size() == recognizedCrossesMoves.size());

    sortByLoc(staticStartPosMoves);

    for(size_t i = 0; i < staticStartPosMoves.size(); i++) {
      if(staticStartPosMoves[i].loc != recognizedCrossesMoves[i].loc || staticStartPosMoves[i].pla != recognizedCrossesMoves[i].pla) {
        randomized = true;
        break;
      }
    }

    if (remainingMoves != nullptr) {
      for (const auto recognizedMove : recognizedCrossesMoves) {
        bool removed = false;
        for(auto it = remainingMoves->begin(); it != remainingMoves->end(); ++it) {
          if (it->loc == recognizedMove.loc && it->pla == recognizedMove.pla) {
            remainingMoves->erase(it);
            removed = true;
            break;
          }
        }
        assert(removed);
      }
    }

    result = expectedStartPos;
  };

  switch (recognizedCrossesMoves.size()) {
    case 4:
      detectRandomization(START_POS_CROSS);
      break;
    case 8:
      detectRandomization(START_POS_CROSS_2);
      break;
    case 16:
      detectRandomization(START_POS_CROSS_4);
      break;
    default:;
      break;
  }

  return result;
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

const Hash128 Rules::ZOBRIST_BUTTON_HASH =   //Based on sha256 hash of Rules::ZOBRIST_BUTTON_HASH
  Hash128(0xb8b914c9234ece84ULL, 0x3d759cddebe29c14ULL);

const Hash128 Rules::ZOBRIST_FRIENDLY_PASS_OK_HASH =   //Based on sha256 hash of Rules::ZOBRIST_FRIENDLY_PASS_OK_HASH
  Hash128(0x0113655998ef0a25ULL, 0x99c9d04ecd964874ULL);

const Hash128 Rules::ZOBRIST_DOTS_GAME_HASH =
  Hash128(0xcdbfab9c91da83a9ULL, 0x6c2f198b2742181full);

const Hash128 Rules::ZOBRIST_DOTS_CAPTURE_EMPTY_BASES_HASH =
  Hash128(0x469afde424e960deull, 0x59f6138cebc753afull);

