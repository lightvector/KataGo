#ifndef RULES_H
#define RULES_H

#include "../core/global.h"

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
  ~Rules();

  static set<string> koRuleStrings();
  static set<string> scoringRuleStrings();
  static int parseKoRule(const string& s);
  static int parseScoringRule(const string& s);
  static string writeKoRule(int koRule);
  static string writeScoringRule(int scoringRule);
};

#endif
