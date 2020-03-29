#ifndef DATAIO_HOMEDATA_H_
#define DATAIO_HOMEDATA_H_

#include "../core/global.h"

namespace HomeData {
  //Returns directory for reading files that may have been installed as defaults to
  //command line arguments, in order of preference. May throw StringError if filesystem access fails.
  std::vector<std::string> getDefaultFilesDirs();
  //A version that doesn't access the file system, intended for help messages, and should never fail.
  std::string getDefaultFilesDirForHelpMessage();

  //Returns a directory suitable for writing data that KataGo generates automatically, such as auto-tuning data.
  //May throw StringError if filesystem access fails.
  //If makeDir is true, will attempt to create the directory if it doesn't exist.
  std::string getHomeDataDir(bool makeDir);
}

#endif //DATAIO_HOMEDATA_H_
