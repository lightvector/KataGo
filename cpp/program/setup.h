#ifndef PROGRAM_SETUP_H_
#define PROGRAM_SETUP_H_

#include "../core/config_parser.h"
#include "../core/global.h"
#include "../search/asyncbot.h"

//Some bits of initialization and main function logic shared between various programs
namespace Setup {

  void initializeSession(ConfigParser& cfg);

  std::vector<NNEvaluator*> initializeNNEvaluators(
    const std::vector<std::string>& nnModelNames,
    const std::vector<std::string>& nnModelFiles,
    ConfigParser& cfg,
    Logger& logger,
    Rand& seedRand,
    int maxConcurrentEvals,
    int defaultNNXLen,
    int defaultNNYLen,
    int forcedSymmetry //-1 if not forcing a symmetry
  );

  //Loads search parameters for bot from config, by bot idx.
  //Fails if no parameters are found.
  std::vector<SearchParams> loadParams(
    ConfigParser& cfg
  );

  Player parseReportAnalysisWinrates(
    ConfigParser& cfg, Player defaultPerspective
  );
}

#endif  // PROGRAM_SETUP_H_
