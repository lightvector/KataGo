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

  bool multiStoneSuicideLegal;
  float komi;

  Rules();
  Rules(int koRule, int scoringRule, bool multiStoneSuicideLegal, float komi);
  ~Rules();

  bool operator==(const Rules& other) const;
  bool operator!=(const Rules& other) const;

  static Rules getTrompTaylorish();
  static Rules getSimpleTerritory();

  static std::set<std::string> koRuleStrings();
  static std::set<std::string> scoringRuleStrings();
  static int parseKoRule(const std::string& s);
  static int parseScoringRule(const std::string& s);
  static std::string writeKoRule(int koRule);
  static std::string writeScoringRule(int scoringRule);

  static bool komiIsIntOrHalfInt(float komi);

  static bool tryParseRules(const std::string& str, Rules& buf);

  friend std::ostream& operator<<(std::ostream& out, const Rules& rules);
  std::string toString() const;

  static const Hash128 ZOBRIST_KO_RULE_HASH[4];
  static const Hash128 ZOBRIST_SCORING_RULE_HASH[2];
  static const Hash128 ZOBRIST_MULTI_STONE_SUICIDE_HASH;

};

#endif  // GAME_RULES_H_
