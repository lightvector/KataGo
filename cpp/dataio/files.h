#ifndef DATAIO_FILES_H_
#define DATAIO_FILES_H_

#include "../core/global.h"

namespace FileHelpers {
  void collectSgfsFromDir(const std::string& dir, std::vector<std::string>& collected);
  void collectSgfsFromDirOrFile(const std::string& dirOrFile, std::vector<std::string>& collected);
  void collectSgfsFromDirs(const std::vector<std::string>& dirs, std::vector<std::string>& collected);
  void collectSgfsFromDirsOrFiles(const std::vector<std::string>& dirsOrFiles, std::vector<std::string>& collected);

  void collectMultiSgfsFromDir(const std::string& dir, std::vector<std::string>& collected);
  void collectMultiSgfsFromDirOrFile(const std::string& dirOrFile, std::vector<std::string>& collected);
  void collectMultiSgfsFromDirs(const std::vector<std::string>& dirs, std::vector<std::string>& collected);
  void collectMultiSgfsFromDirsOrFiles(const std::vector<std::string>& dirsOrFiles, std::vector<std::string>& collected);

  void collectPosesFromDir(const std::string& dir, std::vector<std::string>& collected);
  void collectPosesFromDirOrFile(const std::string& dirOrFile, std::vector<std::string>& collected);
  void collectPosesFromDirs(const std::vector<std::string>& dirs, std::vector<std::string>& collected);
  void collectPosesFromDirsOrFiles(const std::vector<std::string>& dirsOrFiles, std::vector<std::string>& collected);

  void sortNewestToOldest(std::vector<std::string>& files);
}


#endif // DATAIO_FILES_H_
