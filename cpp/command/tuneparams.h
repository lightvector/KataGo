#ifndef COMMAND_TUNEPARAMS_H_
#define COMMAND_TUNEPARAMS_H_

#include "../core/config_parser.h"

namespace TuneParams {
  // Returns the index of the selected dimension within tuneDims[].
  // Throws StringError if cfg's tuneDimension value is not in the allowlist.
  // Propagates ConfigParser's exception if tuneDimension is missing.
  int resolveDimension(ConfigParser& cfg);

  // Self-tests for the dim-resolver. Wired into MainCmds::runtests.
  void runTests();
}

#endif  // COMMAND_TUNEPARAMS_H_
