#ifndef CORE_MAINARGS_H_
#define CORE_MAINARGS_H_

#include "../core/global.h"

namespace MainArgs {
  std::vector<std::string> getCommandLineArgsUTF8(int argc, const char* const* argv);

  void makeCoutAndCerrAcceptUTF8();
}

#endif
