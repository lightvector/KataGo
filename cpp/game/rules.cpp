#include "../game/rules.h"

Rules::Rules() {
  //Defaults if not set - closest match to TT rules
  koRule = KO_POSITIONAL;
  scoringRule = SCORING_AREA;
  multiStoneSuicideLegal = true;
  komi = 7.5f;
}

Rules::~Rules() {
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
