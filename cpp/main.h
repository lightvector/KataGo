#include "core/global.h"
using namespace std;

namespace MainCmds {
  int evalsgf(int argc, const char* const* argv);
  int gatekeeper(int argc, const char* const* argv);
  int gtp(int argc, const char* const* argv);
  int match(int argc, const char* const* argv);
  int selfplay(int argc, const char* const* argv);
  int runtests(int argc, const char* const* argv);
  int runoutputtests(int argc, const char* const* argv);
  int runsearchtests(int argc, const char* const* argv);
  int runsearchtestsv3(int argc, const char* const* argv);

  int writeSearchValueTimeseries(int argc, const char* const* argv);

  int sandbox();
}
