#ifndef CORE_FILEUTILS_H_
#define CORE_FILEUTILS_H_

#include "../core/global.h"

namespace FileUtils {
  bool exists(const std::string& path);

  // Returns whether good() is true on the fstream after attempting to call open on it.
  bool tryOpen(std::ifstream& in, const char* filename, std::ios_base::openmode mode = std::ios_base::in);
  bool tryOpen(std::ofstream& out, const char* filename, std::ios_base::openmode mode = std::ios_base::out);
  bool tryOpen(std::ifstream& in, const std::string& filename, std::ios_base::openmode mode = std::ios_base::in);
  bool tryOpen(std::ofstream& out, const std::string& filename, std::ios_base::openmode mode = std::ios_base::out);
  // Raises exception unless file open was successful
  void open(std::ifstream& in, const char* filename, std::ios_base::openmode mode = std::ios_base::in);
  void open(std::ofstream& out, const char* filename, std::ios_base::openmode mode = std::ios_base::out);
  void open(std::ifstream& in, const std::string& filename, std::ios_base::openmode mode = std::ios_base::in);
  void open(std::ofstream& out, const std::string& filename, std::ios_base::openmode mode = std::ios_base::out);

  void loadFileIntoString(const std::string& filename, const std::string& expectedSha256, std::string& buf);
  void loadFileIntoString(const std::string& filename, const std::string& expectedSha256, std::string& buf, std::string* actualSha256Buf);
  void uncompressAndLoadFileIntoString(const std::string& filename, const std::string& expectedSha256, std::string& buf);
  void uncompressAndLoadFileIntoString(const std::string& filename, const std::string& expectedSha256, std::string& buf, std::string* actualSha256Buf);

  bool isDirectory(const std::string& filename);
  bool tryRemoveFile(const std::string& filename);

  std::string weaklyCanonical(const std::string& path);

  bool tryRename(const std::string& src, const std::string& dst);
  void rename(const std::string& src, const std::string& dst);

  // Read entire file whole
  std::string readFile(const char* filename);
  std::string readFile(const std::string& filename);
  std::string readFileBinary(const char* filename);
  std::string readFileBinary(const std::string& filename);

  // Read file into separate lines, using the specified delimiter character(s).
  // The delimiter characters are NOT included.
  std::vector<std::string> readFileLines(const char* filename, char delimiter);
  std::vector<std::string> readFileLines(const std::string& filename, char delimiter);

  // Return list of filenames of files in a directory
  std::vector<std::string> listFiles(const std::string& dirname);

  // Recursively walk a directory and find all the files that match fileFilter.
  // fileFilter receives just the file name and not the full path, but collected contains the paths.
  void collectFiles(const std::string& dirname, std::function<bool(const std::string&)> fileFilter, std::vector<std::string>& collected);
}


#endif // CORE_FILEUTILS_H_
