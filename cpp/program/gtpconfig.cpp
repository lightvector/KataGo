#include "../program/gtpconfig.h"

using namespace std;

static const string gtpBase = R"%%(

# Logs------------------------------------------------------------------------------------

# Where to output log?
logDir = gtp_logs    # Each run of KataGo will log to a separate file in this dir
# logFile = gtp.log  # Use this instead of logDir to just specify a single file directly

# Logging options
logAllGTPCommunication = true
logSearchInfo = true
logToStderr = false

# Analysis------------------------------------------------------------------------------------

# Configure the maximum length of analysis printed out by lz-analyze and other places.
# Controls the number of moves after the first move in a variation.
# analysisPVLen = 15

# Report winrates for chat and analysis as (BLACK|WHITE|SIDETOMOVE).
# Default is SIDETOMOVE, which is what tools that use LZ probably also expect
# reportAnalysisWinratesAs = SIDETOMOVE

# Uncomment and and set to a positive value to make KataGo explore the top move(s) less deeply and accurately,
# but explore and give evaluations to a greater variety of moves, for analysis (does NOT affect play).
# A value like 0.03 or 0.06 will give various mildly but still noticeably wider searches.
# An extreme value like 1 will distribute many playouts across every move on the board, even very bad moves.
# analysisWideRootNoise = 0.0


# Default rules------------------------------------------------------------------------------------
# See https://lightvector.github.io/KataGo/rules.html for a description of the rules.
# These rules are defaults and can be changed mid-run by several custom GTP commands.
# See https://github.com/lightvector/KataGo/blob/master/docs/GTP_Extensions.md for those commands.

$$KO_RULE

$$SCORING_RULE

$$TAX_RULE

$$MULTI_STONE_SUICIDE

$$BUTTON

$$WHITE_HANDICAP_BONUS

# Bot behavior---------------------------------------------------------------------------------------

# Resignation -------------

# Resignation occurs if for at least resignConsecTurns in a row,
# the winLossUtility (which is on a [-1,1] scale) is below resignThreshold.
allowResignation = true
resignThreshold = -0.90
resignConsecTurns = 3
# Uncomment to make katago not resign close games, behind by fewer than this many points
# resignMinScoreDifference = 10

# Handicap -------------

# Assume that if black makes many moves in a row right at the start of the game, then the game is a handicap game.
# This is necessary on some servers and for some GUIs and also when initializing from many SGF files, which may
# set up a handicap games using repeated GTP "play" commands for black rather than GTP "place_free_handicap" commands.
# However, it may also lead to incorrect understanding of komi if whiteHandicapBonus is used and a server does NOT
# have such a practice.
# Defaults to true! Uncomment and set to false to disable this behavior.
# assumeMultipleStartingBlackMovesAreHandicap = true

# Makes katago dynamically adjust in handicap or altered-komi games to assume it is stronger or weaker than the opponent
# based on those game settings making sense, greatly improving handicap strength but biasing winrates and scores.
# Does NOT affect analysis (lz-analyze, kata-analyze, used by programs like Lizzie) so analysis remains unbiased.
# Uncomment and set this to 0 to disable this and make KataGo play the same always.
# dynamicPlayoutDoublingAdvantageCapPerOppLead = 0.045

# Instead of a dynamic level, you can uncomment this and set this to a value from -3.0 to 3.0 to set KataGo's aggression to a FIXED level.
# DOES affect analysis (lz-analyze, kata-analyze, used by programs like Lizzie).
# Negative makes KataGo behave as if it is much weaker than the opponent, preferring to play defensively.
# Positive makes KataGo behave as if it is much stronger than the opponent, prefering to play aggressively or even overplay slightly.
# If this and "dynamicPlayoutDoublingAdvantageCapPerOppLead" are BOTH set then dynamic will be used for all games and this fixed
# value will be used for analysis tools.
# playoutDoublingAdvantage = 0.0

# Uncommenting one of these will enforce that the FIXED playoutDoublingAdvantage will only apply when KataGo plays the specified color
# and will be negated when playing the opposite color.
# playoutDoublingAdvantagePla = BLACK
# playoutDoublingAdvantagePla = WHITE

# Misc Behavior --------------------

# Uncomment and set to true to make KataGo avoid a particular joseki that some KataGo nets misevaluate,
# and also to improve opening diversity versus some particular other bots that like to play it all the time.
# avoidMYTDaggerHack = false

# Search limits-----------------------------------------------------------------------------------

# If provided, limit maximum number of root visits per search to this much. (With tree reuse, visits do count earlier search)
$$MAX_VISITS
# If provided, limit maximum number of new playouts per search to this much. (With tree reuse, playouts do not count earlier search)
$$MAX_PLAYOUTS
# If provided, cap search time at this many seconds (search will still try to follow GTP time controls)
$$MAX_TIME

# Ponder on the opponent's turn?
$$PONDERING

# Number of seconds to buffer for lag for GTP time controls
lagBuffer = 1.0

# Number of threads to use in search
numSearchThreads = $$NUM_SEARCH_THREADS

# Play a little faster if the opponent is passing, for friendliness
searchFactorAfterOnePass = 0.50
searchFactorAfterTwoPass = 0.25
# Play a little faster if super-winning, for friendliness
searchFactorWhenWinning = 0.40
searchFactorWhenWinningThreshold = 0.95

# GPU Settings-------------------------------------------------------------------------------

# Maximum number of positions to send to a single GPU at once.
# The default value here is roughly equal to numSearchThreads, but you can specify it manually
# if you are running out of memory, or if you are using multiple GPUs that expect to split
# up the work.
# nnMaxBatchSize = <integer>

# Cache up to (2 ** this) many neural net evaluations in case of transpositions in the tree.
# Uncomment and edit to change if you want to adjust a major component of KataGo's RAM usage.
nnCacheSizePowerOfTwo = $$NN_CACHE_SIZE_POWER_OF_TWO

# Size of mutex pool for nnCache is (2 ** this).
nnMutexPoolSizePowerOfTwo = $$NN_MUTEX_POOL_SIZE_POWER_OF_TWO

$$MULTIPLE_GPUS


# Internal params------------------------------------------------------------------------------
# Uncomment and edit any of the below values to change them from their default.

# How big to make the mutex pool for search synchronization
# mutexPoolSize = 16384

)%%";


string GTPConfig::makeConfig(
  const Rules& rules,
  int64_t maxVisits,
  int64_t maxPlayouts,
  double maxTime,
  double maxPonderTime,
  std::vector<int> deviceIdxs,
  int nnCacheSizePowerOfTwo,
  int nnMutexPoolSizePowerOfTwo,
  int numSearchThreads
) {
  string config = gtpBase;
  auto replace = [&](const string& key, const string& replacement) {
    size_t pos = config.find(key);
    assert(pos != string::npos);
    config.replace(pos, key.size(), replacement);
  };

  if(rules.koRule == Rules::KO_SIMPLE)      replace("$$KO_RULE", "koRule = SIMPLE  # options: SIMPLE, POSITIONAL, SITUATIONAL");
  else if(rules.koRule == Rules::KO_POSITIONAL)  replace("$$KO_RULE", "koRule = POSITIONAL  # options: SIMPLE, POSITIONAL, SITUATIONAL");
  else if(rules.koRule == Rules::KO_SITUATIONAL) replace("$$KO_RULE", "koRule = SITUATIONAL  # options: SIMPLE, POSITIONAL, SITUATIONAL");
  else if(rules.koRule == Rules::KO_SPIGHT) replace("$$KO_RULE", "koRule = SPIGHT  # options: SIMPLE, POSITIONAL, SITUATIONAL");
  else { ASSERT_UNREACHABLE; }

  if(rules.scoringRule == Rules::SCORING_AREA)            replace("$$SCORING_RULE", "scoringRule = AREA  # options: AREA, TERRITORY");
  else if(rules.scoringRule == Rules::SCORING_TERRITORY)  replace("$$SCORING_RULE", "scoringRule = TERRITORY  # options: AREA, TERRITORY");
  else { ASSERT_UNREACHABLE; }

  if(rules.taxRule == Rules::TAX_NONE)      replace("$$TAX_RULE", "taxRule = NONE  # options: NONE, SEKI, ALL");
  else if(rules.taxRule == Rules::TAX_SEKI) replace("$$TAX_RULE", "taxRule = SEKI  # options: NONE, SEKI, ALL");
  else if(rules.taxRule == Rules::TAX_ALL)  replace("$$TAX_RULE", "taxRule = ALL  # options: NONE, SEKI, ALL");
  else { ASSERT_UNREACHABLE; }

  if(rules.multiStoneSuicideLegal) replace("$$MULTI_STONE_SUICIDE", "multiStoneSuicideLegal = true");
  else                             replace("$$MULTI_STONE_SUICIDE", "multiStoneSuicideLegal = false");

  if(rules.hasButton) replace("$$BUTTON", "hasButton = true");
  else                replace("$$BUTTON", "hasButton = false");

  if(rules.whiteHandicapBonusRule == Rules::WHB_ZERO)              replace("$$WHITE_HANDICAP_BONUS", "whiteHandicapBonus = 0  # options: 0, N, N-1");
  else if(rules.whiteHandicapBonusRule == Rules::WHB_N)            replace("$$WHITE_HANDICAP_BONUS", "whiteHandicapBonus = N  # options: 0, N, N-1");
  else if(rules.whiteHandicapBonusRule == Rules::WHB_N_MINUS_ONE)  replace("$$WHITE_HANDICAP_BONUS", "whiteHandicapBonus = N-1  # options: 0, N, N-1");
  else { ASSERT_UNREACHABLE; }

  if(maxVisits < ((int64_t)1 << 50)) replace("$$MAX_VISITS", "maxVisits = " + Global::int64ToString(maxVisits));
  else                               replace("$$MAX_VISITS", "#maxVisits = 500");
  if(maxPlayouts < ((int64_t)1 << 50)) replace("$$MAX_PLAYOUTS", "maxPlayouts = " + Global::int64ToString(maxPlayouts));
  else                                 replace("$$MAX_PLAYOUTS", "#maxPlayouts = 300");
  if(maxTime < 1e20)                   replace("$$MAX_TIME", "maxTime = " + Global::doubleToString(maxTime));
  else                                 replace("$$MAX_TIME", "#maxTime = 10");

  if(maxPonderTime <= 0)               replace("$$PONDERING", "ponderingEnabled = false\n#maxTimePondering = 60");
  else if(maxPonderTime < 1e20)        replace("$$PONDERING", "ponderingEnabled = true\nmaxTimePondering = " + Global::doubleToString(maxPonderTime));
  else                                 replace("$$PONDERING", "ponderingEnabled = true\n#maxTimePondering = 60");

  replace("$$NUM_SEARCH_THREADS", Global::intToString(numSearchThreads));
  replace("$$NN_CACHE_SIZE_POWER_OF_TWO", Global::intToString(nnCacheSizePowerOfTwo));
  replace("$$NN_MUTEX_POOL_SIZE_POWER_OF_TWO", Global::intToString(nnMutexPoolSizePowerOfTwo));

  if(deviceIdxs.size() <= 0) {
    replace("$$MULTIPLE_GPUS", "");
  }
  else {
    string replacement = "";
    replacement += "numNNServerThreadsPerModel = " + Global::intToString(deviceIdxs.size()) + "\n";

    for(int i = 0; i<deviceIdxs.size(); i++) {
#ifdef USE_CUDA_BACKEND
      replacement += "cudaDeviceToUseThread" + Global::intToString(i) + " = " + Global::intToString(deviceIdxs[i]) + "\n";
#endif
#ifdef USE_OPENCL_BACKEND
      replacement += "openclDeviceToUseThread" + Global::intToString(i) + " = " + Global::intToString(deviceIdxs[i]) + "\n";
#endif
    }
    replace("$$MULTIPLE_GPUS", replacement);
  }

  return config;
}
