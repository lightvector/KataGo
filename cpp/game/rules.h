#ifndef GAME_RULES_H_
#define GAME_RULES_H_

#include "../core/global.h"
#include "../core/hash.h"

#include "../external/nlohmann_json/json.hpp"
using json = nlohmann::json;

struct Rules {

  static const int KO_SIMPLE = 0;
  static const int KO_POSITIONAL = 1;
  static const int KO_SITUATIONAL = 2;
  static const int KO_SPIGHT = 3;
  int koRule;

  static const int SCORING_AREA = 0;
  static const int SCORING_TERRITORY = 1;
  int scoringRule;

  static const int TAX_NONE = 0;
  static const int TAX_SEKI = 1;
  static const int TAX_ALL = 2;
  int taxRule;

  bool multiStoneSuicideLegal;
  bool hasButton;

  static const int WHB_ZERO = 0;
  static const int WHB_N = 1;
  static const int WHB_N_MINUS_ONE = 2;
  int whiteHandicapBonusRule;

  float komi;
  //Min and max acceptable komi in various places involving user input validation
  static constexpr float MIN_USER_KOMI = -150.0f;
  static constexpr float MAX_USER_KOMI = 150.0f;

  Rules();
  Rules(int koRule, int scoringRule, int taxRule, bool multiStoneSuicideLegal, bool hasButton, int whiteHandicapBonusRule, float komi);
  ~Rules();

  bool operator==(const Rules& other) const;
  bool operator!=(const Rules& other) const;

  bool equalsIgnoringKomi(const Rules& other) const;
  bool gameResultWillBeInteger() const;

  static Rules getTrompTaylorish();
  static Rules getSimpleTerritory();

  static std::set<std::string> koRuleStrings();
  static std::set<std::string> scoringRuleStrings();
  static std::set<std::string> taxRuleStrings();
  static std::set<std::string> whiteHandicapBonusRuleStrings();
  static int parseKoRule(const std::string& s);
  static int parseScoringRule(const std::string& s);
  static int parseTaxRule(const std::string& s);
  static int parseWhiteHandicapBonusRule(const std::string& s);
  static std::string writeKoRule(int koRule);
  static std::string writeScoringRule(int scoringRule);
  static std::string writeTaxRule(int taxRule);
  static std::string writeWhiteHandicapBonusRule(int whiteHandicapBonusRule);

  static bool komiIsIntOrHalfInt(float komi);

  static Rules parseRules(const std::string& str);
  static Rules parseRulesWithoutKomi(const std::string& str, float komi);
  static bool tryParseRules(const std::string& str, Rules& buf);
  static bool tryParseRulesWithoutKomi(const std::string& str, Rules& buf, float komi);

  static Rules updateRules(const std::string& key, const std::string& value, Rules priorRules);

  friend std::ostream& operator<<(std::ostream& out, const Rules& rules);
  std::string toString() const;
  std::string toStringNoKomi() const;
  std::string toStringNoKomiMaybeNice() const;
  json toJson() const;
  std::string toJsonString() const;
  std::string toJsonStringNoKomi() const;
  std::string toJsonStringNoKomiMaybeOmitStuff() const;

  static const Hash128 ZOBRIST_KO_RULE_HASH[4];
  static const Hash128 ZOBRIST_SCORING_RULE_HASH[2];
  static const Hash128 ZOBRIST_TAX_RULE_HASH[3];
  static const Hash128 ZOBRIST_MULTI_STONE_SUICIDE_HASH;
  static const Hash128 ZOBRIST_BUTTON_HASH;
};

#endif  // GAME_RULES_H_
