#include "../dataio/files.h"

#include "../core/fileutils.h"

#include <ghc/filesystem.hpp>

using namespace std;
namespace gfs = ghc::filesystem;

static const string sgfSuffix = ".sgf";
static const string sgfSuffix2 = ".SGF";
static bool sgfFilter(const string& name) {
  return Global::isSuffix(name,sgfSuffix) || Global::isSuffix(name,sgfSuffix2);
}

void FileHelpers::collectSgfsFromDir(const std::string& dir, std::vector<std::string>& collected) {
  FileUtils::collectFiles(dir, &sgfFilter, collected);
}

void FileHelpers::collectSgfsFromDirOrFile(const std::string& dirOrFile, std::vector<std::string>& collected) {
  try {
    if(gfs::exists(dirOrFile) && !gfs::is_directory(dirOrFile)) {
      if(sgfFilter(dirOrFile))
        collected.push_back(dirOrFile);
      else {
        cerr << "Error collecting sgf files: File does not end in .sgf or .SGF: " << dirOrFile << endl;
        throw StringError(string("Error collecting sgf files: File does not end in .sgf or .SGF: ") + dirOrFile);
      }
      return;
    }
  }
  catch(const gfs::filesystem_error& e) {
    cerr << "Error recursively collecting files: " << e.what() << endl;
    throw StringError(string("Error recursively collecting files: ") + e.what());
  }
  FileHelpers::collectSgfsFromDir(dirOrFile, collected);
}

void FileHelpers::collectSgfsFromDirs(const std::vector<std::string>& dirs, std::vector<std::string>& collected) {
  for(int i = 0; i<dirs.size(); i++) {
    string trimmed = Global::trim(dirs[i]);
    if(trimmed.size() <= 0)
      continue;
    if(gfs::exists(dirs[i]))
      collectSgfsFromDir(dirs[i], collected);
    else
      collectSgfsFromDir(trimmed, collected);
  }
}

void FileHelpers::collectSgfsFromDirsOrFiles(const std::vector<std::string>& dirsOrFiles, std::vector<std::string>& collected) {
  for(int i = 0; i<dirsOrFiles.size(); i++) {
    string trimmed = Global::trim(dirsOrFiles[i]);
    if(trimmed.size() <= 0)
      continue;
    if(gfs::exists(dirsOrFiles[i]))
      collectSgfsFromDirOrFile(dirsOrFiles[i], collected);
    else
      collectSgfsFromDirOrFile(trimmed, collected);
  }
}

void FileHelpers::sortNewestToOldest(std::vector<std::string>& files) {
  vector<std::pair<string, gfs::file_time_type>> filesWithTime;
  for(size_t i = 0; i<files.size(); i++)
    filesWithTime.push_back(std::make_pair(files[i], gfs::last_write_time(files[i])));

  std::sort(
    filesWithTime.begin(),
    filesWithTime.end(),
    [](const std::pair<string, gfs::file_time_type>& a, std::pair<string, gfs::file_time_type>& b) -> bool {
      return a.second > b.second;
    }
  );

  for(size_t i = 0; i<files.size(); i++)
    files[i] = filesWithTime[i].first;
}
