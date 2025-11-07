#ifndef GAME_RULES_H_
#define GAME_RULES_H_

#include "common.h"
#include "../core/global.h"
#include "../core/hash.h"
#include "../core/rand.h"

#include "../external/nlohmann_json/json.hpp"

struct Rules {
  const static Rules DEFAULT_DOTS;
  const static Rules DEFAULT_GO;

  static constexpr int START_POS_EMPTY = 0;
  static constexpr int START_POS_CROSS = 1;
  static constexpr int START_POS_CROSS_2 = 2;
  static constexpr int START_POS_CROSS_4 = 3;
  static constexpr int START_POS_CUSTOM = 4;
  int startPos;

  // Enables random shuffling of start pos. Currently, it works only for CROSS_4
  bool startPosIsRandom;

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

  static const int WHB_ZERO = 0;
  static const int WHB_N = 1;
  static const int WHB_N_MINUS_ONE = 2;
  int whiteHandicapBonusRule;

  float komi;
  //Min and max acceptable komi in various places involving user input validation
  static constexpr float MIN_USER_KOMI = -150.0f;
  static constexpr float MAX_USER_KOMI = 150.0f;

  bool isDots;

  bool dotsCaptureEmptyBases;
  bool dotsFreeCapturedDots; // TODO: Implement later
  bool multiStoneSuicideLegal; // Works as just suicide in Dots Game
  bool hasButton;
  //Mostly an informational value - doesn't affect the actual implemented rules, but GTP or Analysis may, at a
  //high level, use this info to adjust passing behavior - whether it's okay to pass without capturing dead stones.
  //Only relevant for area scoring.
  bool friendlyPassOk;

  Rules();
  Rules(bool initIsDots, int startPos, bool startPosIsRandom, bool dotsCaptureEmptyBases, bool dotsFreeCapturedDots);
  explicit Rules(bool initIsDots);
  Rules(
    int koRule,
    int scoringRule,
    int taxRule,
    bool multiStoneSuicideLegal,
    bool hasButton,
    int whiteHandicapBonusRule,
    bool friendlyPassOk,
    float komi
  );
  ~Rules();

  bool operator==(const Rules& other) const;
  bool operator!=(const Rules& other) const;

  [[nodiscard]] bool equalsIgnoringSgfDefinedProps(const Rules& other) const;
  [[nodiscard]] bool equals(const Rules& other, bool ignoreSgfDefinedProps) const;
  [[nodiscard]] bool gameResultWillBeInteger() const;

  static Rules getDefault(bool isDots);
  static Rules getDefaultOrTrompTaylorish(bool isDots);
  static Rules getTrompTaylorish();
  static Rules getSimpleTerritory();

  static std::set<std::string> koRuleStrings();
  static std::set<std::string> scoringRuleStrings();
  static std::set<std::string> taxRuleStrings();
  static std::set<std::string> whiteHandicapBonusRuleStrings();
  static int parseStartPos(const std::string& s);
  static int parseKoRule(const std::string& s);
  static int parseScoringRule(const std::string& s);
  static int parseTaxRule(const std::string& s);
  static int parseWhiteHandicapBonusRule(const std::string& s);
  static std::string writeStartPosRule(int startPosRule);
  static std::string writeKoRule(int koRule);
  static std::string writeScoringRule(int scoringRule);
  static std::string writeTaxRule(int taxRule);
  static std::string writeWhiteHandicapBonusRule(int whiteHandicapBonusRule);

  static bool komiIsIntOrHalfInt(float komi);
  static std::set<std::string> startPosStrings();
  int getNumOfStartPosStones() const;

  static Rules parseRules(const std::string& sOrig, bool isDots = false);
  static Rules parseRulesWithoutKomi(const std::string& sOrig, float komi, bool isDots = false);
  static bool tryParseRules(const std::string& sOrig, Rules& buf, bool isDots);
  static bool tryParseRulesWithoutKomi(const std::string& sOrig, Rules& buf, float komi, bool isDots);

  static Rules updateRules(const std::string& key, const std::string& value, Rules priorRules);

  static std::vector<Move> generateStartPos(int startPos, Rand* rand, int x_size, int y_size);
  /**
   * @param placementMoves placement moves that we are trying to recognize.
   * @param x_size size of field
   * @param y_size size of field
   * @param emptyIfFailed returns empty start pos if recognition is failed. It's useful for detecting start pos from SGF when handicap stones are placed
   * @param randomized if we recognize a start pos, but it doesn't match the strict position, set it up to `true`
   */
  static int tryRecognizeStartPos(
    const std::vector<Move>& placementMoves,
    int x_size,
    int y_size,
    bool emptyIfFailed,
    bool& randomized);

  friend std::ostream& operator<<(std::ostream& out, const Rules& rules);
  std::string toString(bool includeSgfDefinedProperties = true) const;
  std::string toStringNoSgfDefinedPropertiesMaybeNice() const;
  std::string toJsonString() const;
  std::string toJsonStringNoKomi() const;
  std::string toJsonStringNoKomiMaybeOmitStuff() const;
  nlohmann::json toJson() const;
  nlohmann::json toJsonNoKomi() const;
  nlohmann::json toJsonNoKomiMaybeOmitStuff() const;

  static const Hash128 ZOBRIST_KO_RULE_HASH[4];
  static const Hash128 ZOBRIST_SCORING_RULE_HASH[2];
  static const Hash128 ZOBRIST_TAX_RULE_HASH[3];
  static const Hash128 ZOBRIST_MULTI_STONE_SUICIDE_HASH;
  static const Hash128 ZOBRIST_BUTTON_HASH;
  static const Hash128 ZOBRIST_FRIENDLY_PASS_OK_HASH;
  static const Hash128 ZOBRIST_DOTS_GAME_HASH;
  static const Hash128 ZOBRIST_DOTS_CAPTURE_EMPTY_BASES_HASH;

private:
  Rules(
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
  );

  static inline std::map<int, std::string> startPosIdToName;
  static inline std::map<std::string, int> startPosNameToId;

  static void initializeIfNeeded() {
    if (startPosIdToName.empty()) {
      startPosIdToName[START_POS_EMPTY] = "EMPTY";
      startPosIdToName[START_POS_CROSS] = "CROSS";
      startPosIdToName[START_POS_CROSS_2] = "CROSS_2";
      startPosIdToName[START_POS_CROSS_4] = "CROSS_4";
      startPosNameToId["EMPTY"] = START_POS_EMPTY;
      startPosNameToId["CROSS"] = START_POS_CROSS;
      startPosNameToId["CROSS_2"] = START_POS_CROSS_2;
      startPosNameToId["CROSS_4"] = START_POS_CROSS_4;
    }
  }

  static void addCross(int x, int y, int x_size, bool rotate90, std::vector<Move>& moves);

  nlohmann::json toJsonHelper(bool omitKomi, bool omitDefaults) const;
};

#endif  // GAME_RULES_H_
