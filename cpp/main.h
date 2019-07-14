#include "core/global.h"

namespace MainCmds {
  int evalsgf(int argc, const char* const* argv);
  int gatekeeper(int argc, const char* const* argv);
  int gtp(int argc, const char* const* argv);
  int tuner(int argc, const char* const* argv);
  int match(int argc, const char* const* argv);
  int matchauto(int argc, const char* const* argv);
  int selfplay(int argc, const char* const* argv);
  int runtests(int argc, const char* const* argv);
  int runnnlayertests(int argc, const char* const* argv);
  int runnnontinyboardtest(int argc, const char* const* argv);
  int runoutputtests(int argc, const char* const* argv);
  int runsearchtests(int argc, const char* const* argv);
  int runsearchtestsv3(int argc, const char* const* argv);
  int runselfplayinittests(int argc, const char* const* argv);
  int runnnonmanyposestest(int argc, const char* const* argv);

  int lzcost(int argc, const char* const* argv);
  int demoplay(int argc, const char* const* argv);

  int sandbox();
}

namespace Version {
  std::string getKataGoVersion();
  std::string getKataGoVersionForHelp();
  std::string getGitRevision();
}
