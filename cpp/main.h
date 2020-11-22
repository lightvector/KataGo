#include "core/global.h"

namespace MainCmds {
  int analysis(int argc, const char* const* argv);
  int benchmark(int argc, const char* const* argv);
  int contribute(int argc, const char* const* argv);
  int evalsgf(int argc, const char* const* argv);
  int gatekeeper(int argc, const char* const* argv);
  int genconfig(int argc, const char* const* argv, const char* firstCommand);
  int gtp(int argc, const char* const* argv);
  int tuner(int argc, const char* const* argv);
  int match(int argc, const char* const* argv);
  int matchauto(int argc, const char* const* argv);
  int selfplay(int argc, const char* const* argv);
  int runtests(int argc, const char* const* argv);
  int runnnlayertests(int argc, const char* const* argv);
  int runnnontinyboardtest(int argc, const char* const* argv);
  int runnnsymmetriestest(int argc, const char* const* argv);
  int runoutputtests(int argc, const char* const* argv);
  int runsearchtests(int argc, const char* const* argv);
  int runsearchtestsv3(int argc, const char* const* argv);
  int runsearchtestsv8(int argc, const char* const* argv);
  int runselfplayinittests(int argc, const char* const* argv);
  int runsekitrainwritetests(int argc, const char* const* argv);
  int runnnonmanyposestest(int argc, const char* const* argv);
  int runownershiptests(int argc, const char* const* argv);

  int samplesgfs(int argc, const char* const* argv);
  int dataminesgfs(int argc, const char* const* argv);

  int trystartposes(int argc, const char* const* argv);
  int viewstartposes(int argc, const char* const* argv);

  int lzcost(int argc, const char* const* argv);
  int demoplay(int argc, const char* const* argv);
  int printclockinfo(int argc, const char* const* argv);

  int sandbox();
}

namespace Version {
  std::string getKataGoVersion();
  std::string getKataGoVersionForHelp();
  std::string getKataGoVersionFullInfo();
  std::string getGitRevision();
}
