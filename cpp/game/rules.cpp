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
