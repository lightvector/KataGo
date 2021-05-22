#ifndef CORE_FILEUTILS_H_
#define CORE_FILEUTILS_H_

#include "../core/global.h"

namespace FileUtils {
  void loadFileIntoString(const std::string& filename, const std::string& expectedSha256, std::string& buf);
  void uncompressAndLoadFileIntoString(const std::string& filename, const std::string& expectedSha256, std::string& buf);
}


#endif // CORE_FILEUTILS_H_
