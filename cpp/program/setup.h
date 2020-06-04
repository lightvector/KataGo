#ifndef PROGRAM_SETUP_H_
#define PROGRAM_SETUP_H_

#include "../core/config_parser.h"
#include "../core/global.h"
#include "../search/asyncbot.h"

//Some bits of initialization and main function logic shared between various programs
namespace Setup {

  void initializeSession(ConfigParser& cfg);

  enum setup_for_t {
    SETUP_FOR_GTP,
    SETUP_FOR_BENCHMARK,
    SETUP_FOR_MATCH,
    SETUP_FOR_ANALYSIS,
    SETUP_FOR_OTHER,
    SETUP_FOR_DISTRIBUTED
  };

  NNEvaluator* initializeNNEvaluator(
    const std::string& nnModelNames,
    const std::string& nnModelFiles,
    ConfigParser& cfg,
    Logger& logger,
    Rand& seedRand,
    int maxConcurrentEvals,
    int defaultNNXLen,
    int defaultNNYLen,
    int defaultMaxBatchSize,
    setup_for_t setupFor
  );

  std::vector<NNEvaluator*> initializeNNEvaluators(
    const std::vector<std::string>& nnModelNames,
    const std::vector<std::string>& nnModelFiles,
    ConfigParser& cfg,
    Logger& logger,
    Rand& seedRand,
    int maxConcurrentEvals,
    int defaultNNXLen,
    int defaultNNYLen,
    int defaultMaxBatchSize,
    setup_for_t setupFor
  );

  //Loads search parameters for bot from config, by bot idx.
  //Fails if no parameters are found.
  std::vector<SearchParams> loadParams(
    ConfigParser& cfg
  );
  SearchParams loadSingleParams(
    ConfigParser& cfg
  );

  Player parseReportAnalysisWinrates(
    ConfigParser& cfg, Player defaultPerspective
  );

  //Komi is just set to 7.5 and is not read in from cfg
  Rules loadSingleRulesExceptForKomi(
    ConfigParser& cfg
  );

  std::string loadHomeDataDirOverride(
    ConfigParser& cfg
  );

  //Get sets of options that are mutually exclusive. Intended for use in configParser
  std::vector<std::pair<std::set<std::string>,std::set<std::string>>> getMutexKeySets();
}

#endif  // PROGRAM_SETUP_H_
