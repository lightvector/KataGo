#include "../neuralnet/nninputs.h"

//------------------------
#include "../core/using.h"
//------------------------

static SGFMetadata makeBasicRankProfile(int inverseRank, bool preAZ) {
  // KGS rating system is pretty reasonable, so let's use KGS as the source.
  SGFMetadata ret;
  ret.initialized = true;
  ret.inverseBRank = inverseRank;
  ret.inverseWRank = inverseRank;
  ret.bIsHuman = true;
  ret.wIsHuman = true;
  ret.gameRatednessIsUnknown = true;
  ret.tcIsByoYomi = true;
  ret.mainTimeSeconds = 1200;
  ret.periodTimeSeconds = 30;
  ret.byoYomiPeriods = 5;
  if(preAZ)
    ret.gameDate = SimpleDate(2016,9,1);
  else
    ret.gameDate = SimpleDate(2020,3,1);
  ret.source = SGFMetadata::SOURCE_KGS;
  return ret;
}

static SGFMetadata makeHistoricalProProfile(SimpleDate date) {
  SGFMetadata ret;
  ret.initialized = true;
  ret.inverseBRank = 1;
  ret.inverseWRank = 1;
  ret.bIsHuman = true;
  ret.wIsHuman = true;
  ret.tcIsUnknown = true;
  ret.gameDate = date;
  ret.source = SGFMetadata::SOURCE_GOGOD;
  return ret;
}

static SGFMetadata makeModernProProfile(SimpleDate date) {
  SGFMetadata ret;
  ret.initialized = true;
  ret.inverseBRank = 1;
  ret.inverseWRank = 1;
  ret.bIsHuman = true;
  ret.wIsHuman = true;
  ret.tcIsUnknown = true;
  ret.gameDate = date;
  ret.source = SGFMetadata::SOURCE_GO4GO;
  return ret;
}

SGFMetadata SGFMetadata::getProfile(const string& humanSLProfileName) {
  if(Global::isPrefix(humanSLProfileName,"proyear_")) {
    string yearStr = Global::chopPrefix(humanSLProfileName,"proyear_");
    int year;
    bool suc = Global::tryStringToInt(yearStr,year);
    if(suc && year >= 1800 && year <= 2020) {
      return makeHistoricalProProfile(SimpleDate(year,6,1));
    }
    if(suc && year >= 2021 && year <= 2023) {
      return makeModernProProfile(SimpleDate(year,6,1));
    }
  }
  if(Global::isPrefix(humanSLProfileName,"rank_") || Global::isPrefix(humanSLProfileName,"preaz_")) {
    string rankStr;
    bool preAZ;
    if(Global::isPrefix(humanSLProfileName,"rank_")) {
      rankStr = Global::chopPrefix(humanSLProfileName,"rank_");
      preAZ = false;
    }
    else {
      rankStr = Global::chopPrefix(humanSLProfileName,"preaz_");
      preAZ = true;
    }
    if(rankStr == "9d") return makeBasicRankProfile(1,preAZ);
    if(rankStr == "8d") return makeBasicRankProfile(2,preAZ);
    if(rankStr == "7d") return makeBasicRankProfile(3,preAZ);
    if(rankStr == "6d") return makeBasicRankProfile(4,preAZ);
    if(rankStr == "5d") return makeBasicRankProfile(5,preAZ);
    if(rankStr == "4d") return makeBasicRankProfile(6,preAZ);
    if(rankStr == "3d") return makeBasicRankProfile(7,preAZ);
    if(rankStr == "2d") return makeBasicRankProfile(8,preAZ);
    if(rankStr == "1d") return makeBasicRankProfile(9,preAZ);
    if(rankStr == "1k") return makeBasicRankProfile(10,preAZ);
    if(rankStr == "2k") return makeBasicRankProfile(11,preAZ);
    if(rankStr == "3k") return makeBasicRankProfile(12,preAZ);
    if(rankStr == "4k") return makeBasicRankProfile(13,preAZ);
    if(rankStr == "5k") return makeBasicRankProfile(14,preAZ);
    if(rankStr == "6k") return makeBasicRankProfile(15,preAZ);
    if(rankStr == "7k") return makeBasicRankProfile(16,preAZ);
    if(rankStr == "8k") return makeBasicRankProfile(17,preAZ);
    if(rankStr == "9k") return makeBasicRankProfile(18,preAZ);
    if(rankStr == "10k") return makeBasicRankProfile(19,preAZ);
    if(rankStr == "11k") return makeBasicRankProfile(20,preAZ);
    if(rankStr == "12k") return makeBasicRankProfile(21,preAZ);
    if(rankStr == "13k") return makeBasicRankProfile(22,preAZ);
    if(rankStr == "14k") return makeBasicRankProfile(23,preAZ);
    if(rankStr == "15k") return makeBasicRankProfile(24,preAZ);
    if(rankStr == "16k") return makeBasicRankProfile(25,preAZ);
    if(rankStr == "17k") return makeBasicRankProfile(26,preAZ);
    if(rankStr == "18k") return makeBasicRankProfile(27,preAZ);
    if(rankStr == "19k") return makeBasicRankProfile(28,preAZ);
    if(rankStr == "20k") return makeBasicRankProfile(29,preAZ);
  }

  throw StringError("Unknown human SL network profile: " + humanSLProfileName);
}

