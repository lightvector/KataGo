#ifndef NEURALNET_SGFMETADATA_H_
#define NEURALNET_SGFMETADATA_H_

#include <memory>

#include "../core/datetime.h"
#include "../core/hash.h"
#include "../game/board.h"

struct SGFMetadata {
  bool initialized = false;

  int inverseBRank = 0; // KG = 0, 9d = 1, 8d = 2,... 1d = 9, 1k = 10, 2k = 11, ...
  int inverseWRank = 0;
  bool bIsUnranked = false; // KGS "-" rank
  bool wIsUnranked = false;
  bool bRankIsUnknown = false; // No rank in file
  bool wRankIsUnknown = false;
  bool bIsHuman = false;
  bool wIsHuman = false;

  bool gameIsUnrated = false;
  bool gameRatednessIsUnknown = false;

  //One-hot for all things with metadata
  bool tcIsUnknown = false;
  bool tcIsNone = false;
  bool tcIsAbsolute = false;
  bool tcIsSimple = false;
  bool tcIsByoYomi = false;
  bool tcIsCanadian = false;
  bool tcIsFischer = false;

  double mainTimeSeconds = 0.0;
  double periodTimeSeconds = 0.0;
  int byoYomiPeriods = 0;
  int canadianMoves = 0;

  SimpleDate gameDate;

  int source = 0;
  static const int SOURCE_OGS = 1;
  static const int SOURCE_KGS = 2;
  static const int SOURCE_FOX = 3;
  static const int SOURCE_TYGEM = 4;
  static const int SOURCE_GOGOD = 5;
  static const int SOURCE_GO4GO = 6;

  static constexpr int METADATA_INPUT_NUM_CHANNELS = 192;

  bool operator==(const SGFMetadata& other) const;
  bool operator!=(const SGFMetadata& other) const;

  Hash128 getHash(Player nextPlayer) const;

  static void fillMetadataRow(const SGFMetadata* sgfMeta, float* rowMetadata, Player nextPlayer, int boardArea);

  static SGFMetadata getProfile(const std::string& humanSLProfileName);
};

#endif  // NEURALNET_SGFMETADATA_H_
