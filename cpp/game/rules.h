#ifndef RULES_H
#define RULES_H

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

  static Rules getTrompTaylorish();

  static set<string> koRuleStrings();
  static set<string> scoringRuleStrings();
  static int parseKoRule(const string& s);
  static int parseScoringRule(const string& s);
  static string writeKoRule(int koRule);
  static string writeScoringRule(int scoringRule);

  friend ostream& operator<<(ostream& out, const Rules& rules);

  static const Hash128 ZOBRIST_KO_RULE_HASH[4];
  static const Hash128 ZOBRIST_SCORING_RULE_HASH[2];
  static const Hash128 ZOBRIST_MULTI_STONE_SUICIDE_HASH;

};

#endif
