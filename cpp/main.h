#include "core/global.h"
using namespace std;

namespace MainCmds {
  int evalSgf(int argc, const char* const* argv);
  int gtp(int argc, const char* const* argv);
  int match(int argc, const char* const* argv);
  int selfPlay(int argc, const char* const* argv);
  int runTests(int argc, const char* const* argv);
  int runSearchTests(int argc, const char* const* argv);

  int writeSearchValueTimeseries(int argc, const char* const* argv);

  int sandbox();
}
