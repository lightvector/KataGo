#ifndef GAME_RULES_H_
#define GAME_RULES_H_

#include "../core/global.h"
#include "../core/hash.h"

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

  float komi;
  //Min and max acceptable komi in various places involving user input validation
  static constexpr float MIN_USER_KOMI = -150.0f;
  static constexpr float MAX_USER_KOMI = 150.0f;

  Rules();
  Rules(int koRule, int scoringRule, int taxRule, bool multiStoneSuicideLegal, float komi);
  ~Rules();

  bool operator==(const Rules& other) const;
  bool operator!=(const Rules& other) const;

  static Rules getTrompTaylorish();
  static Rules getSimpleTerritory();

  static std::set<std::string> koRuleStrings();
  static std::set<std::string> scoringRuleStrings();
  static std::set<std::string> taxRuleStrings();
  static int parseKoRule(const std::string& s);
  static int parseScoringRule(const std::string& s);
  static int parseTaxRule(const std::string& s);
  static std::string writeKoRule(int koRule);
  static std::string writeScoringRule(int scoringRule);
  static std::string writeTaxRule(int taxRule);

  static bool komiIsIntOrHalfInt(float komi);

  static bool tryParseRules(const std::string& str, Rules& buf);
  static bool tryParseRulesWithoutKomi(const std::string& str, Rules& buf, float komi);

  friend std::ostream& operator<<(std::ostream& out, const Rules& rules);
  std::string toString() const;
  std::string toStringNoKomi() const;
  std::string toJsonString() const;
  std::string toJsonStringNoKomi() const;

  static const Hash128 ZOBRIST_KO_RULE_HASH[4];
  static const Hash128 ZOBRIST_SCORING_RULE_HASH[2];
  static const Hash128 ZOBRIST_TAX_RULE_HASH[3];
  static const Hash128 ZOBRIST_MULTI_STONE_SUICIDE_HASH;
};

#endif  // GAME_RULES_H_
