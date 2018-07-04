#ifndef RULES_H
#define RULES_H

struct Rules {

  static const int KO_SIMPLE = 0;
  static const int KO_POSITIONAL = 1;
  static const int KO_SITUATIONAL = 2;
  int koRule;

  bool multiStoneSuicideLegal;

  double komi;
};

#endif
