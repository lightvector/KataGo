#ifndef GAME_COMMON_H
#define GAME_COMMON_H
#include <cstdint>
#include "../core/global.h"

const std::string DOTS_KEY = "dots";
const std::string DOTS_CAPTURE_EMPTY_BASE_KEY = "dotsCaptureEmptyBase";
const std::string DOTS_CAPTURE_EMPTY_BASES_KEY = "dotsCaptureEmptyBases";
const std::string START_POS_KEY = "startPos";
const std::string START_POS_RANDOM_KEY = "startPosIsRandom";
const std::string START_POSES_KEY = "startPoses";
const std::string START_POSES_ARE_RANDOM_KEY = "startPosesAreRandom";

const std::string BLACK_SCORE_IF_WHITE_GROUNDS_KEY = "blackScoreIfWhiteGrounds";
const std::string WHITE_SCORE_IF_BLACK_GROUNDS_KEY = "whiteScoreIfBlackGrounds";

// Player
typedef int8_t Player;
static constexpr Player P_BLACK = 1;
static constexpr Player P_WHITE = 2;

//Color of a point on the board
typedef int8_t Color;
static constexpr Color C_EMPTY = 0;
static constexpr Color C_BLACK = 1;
static constexpr Color C_WHITE = 2;
static constexpr Color C_WALL = 3;
static constexpr int NUM_BOARD_COLORS = 4;

typedef int8_t State;

//Location of a point on the board
//(x,y) is represented as (x+1) + (y+1)*(x_size+1)
typedef short Loc;

//Simple structure for storing moves. This is a convenient place to define it.
STRUCT_NAMED_PAIR(Loc,loc,Player,pla,Move);

#endif
