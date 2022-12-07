#include "../dataio/files.h"

#include "../core/fileutils.h"

#include <ghc/filesystem.hpp>

using namespace std;
namespace gfs = ghc::filesystem;

static const string sgfSuffix = ".sgf";
static const string sgfSuffix2 = ".SGF";
static const string multiSgfSuffix = ".sgfs";
static const string multiSgfSuffix2 = ".SGFS";
static bool sgfFilter(const string& name) {
  return Global::isSuffix(name,sgfSuffix) || Global::isSuffix(name,sgfSuffix2);
}
static bool multiSgfFilter(const string& name) {
  return Global::isSuffix(name,multiSgfSuffix) || Global::isSuffix(name,multiSgfSuffix2);
}

void FileHelpers::collectSgfsFromDir(const std::string& dir, std::vector<std::string>& collected) {
  FileUtils::collectFiles(dir, &sgfFilter, collected);
}
void FileHelpers::collectMultiSgfsFromDir(const std::string& dir, std::vector<std::string>& collected) {
  FileUtils::collectFiles(dir, &multiSgfFilter, collected);
}

void FileHelpers::collectSgfsFromDirOrFile(const std::string& dirOrFile, std::vector<std::string>& collected) {
  try {
    if(FileUtils::exists(dirOrFile) && !FileUtils::isDirectory(dirOrFile)) {
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
void FileHelpers::collectMultiSgfsFromDirOrFile(const std::string& dirOrFile, std::vector<std::string>& collected) {
  try {
    if(FileUtils::exists(dirOrFile) && !FileUtils::isDirectory(dirOrFile)) {
      if(multiSgfFilter(dirOrFile))
        collected.push_back(dirOrFile);
      else {
        cerr << "Error collecting sgfs files: File does not end in .sgfs or .SGFS: " << dirOrFile << endl;
        throw StringError(string("Error collecting sgfs files: File does not end in .sgfs or .SGFS: ") + dirOrFile);
      }
      return;
    }
  }
  catch(const gfs::filesystem_error& e) {
    cerr << "Error recursively collecting files: " << e.what() << endl;
    throw StringError(string("Error recursively collecting files: ") + e.what());
  }
  FileHelpers::collectMultiSgfsFromDir(dirOrFile, collected);
}

void FileHelpers::collectSgfsFromDirs(const std::vector<std::string>& dirs, std::vector<std::string>& collected) {
  for(int i = 0; i<dirs.size(); i++) {
    string trimmed = Global::trim(dirs[i]);
    if(trimmed.size() <= 0)
      continue;
    if(FileUtils::exists(dirs[i]))
      collectSgfsFromDir(dirs[i], collected);
    else
      collectSgfsFromDir(trimmed, collected);
  }
}
void FileHelpers::collectMultiSgfsFromDirs(const std::vector<std::string>& dirs, std::vector<std::string>& collected) {
  for(int i = 0; i<dirs.size(); i++) {
    string trimmed = Global::trim(dirs[i]);
    if(trimmed.size() <= 0)
      continue;
    if(FileUtils::exists(dirs[i]))
      collectMultiSgfsFromDir(dirs[i], collected);
    else
      collectMultiSgfsFromDir(trimmed, collected);
  }
}

void FileHelpers::collectSgfsFromDirsOrFiles(const std::vector<std::string>& dirsOrFiles, std::vector<std::string>& collected) {
  for(int i = 0; i<dirsOrFiles.size(); i++) {
    string trimmed = Global::trim(dirsOrFiles[i]);
    if(trimmed.size() <= 0)
      continue;
    if(FileUtils::exists(dirsOrFiles[i]))
      collectSgfsFromDirOrFile(dirsOrFiles[i], collected);
    else
      collectSgfsFromDirOrFile(trimmed, collected);
  }
}
void FileHelpers::collectMultiSgfsFromDirsOrFiles(const std::vector<std::string>& dirsOrFiles, std::vector<std::string>& collected) {
  for(int i = 0; i<dirsOrFiles.size(); i++) {
    string trimmed = Global::trim(dirsOrFiles[i]);
    if(trimmed.size() <= 0)
      continue;
    if(FileUtils::exists(dirsOrFiles[i]))
      collectMultiSgfsFromDirOrFile(dirsOrFiles[i], collected);
    else
      collectMultiSgfsFromDirOrFile(trimmed, collected);
  }
}

void FileHelpers::sortNewestToOldest(std::vector<std::string>& files) {
  vector<std::pair<string, gfs::file_time_type>> filesWithTime;
  for(size_t i = 0; i<files.size(); i++)
    filesWithTime.push_back(std::make_pair(files[i], gfs::last_write_time(gfs::u8path(files[i]))));

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
