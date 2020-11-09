#include "../dataio/homedata.h"
#include "../core/os.h"

#ifdef OS_IS_UNIX_OR_APPLE
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include "../external/filesystem-1.3.6/include/ghc/filesystem.hpp"
#endif
#ifdef OS_IS_WINDOWS
#include <windows.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
// Avoid using this for now so as to maintain compatibility with any old windows 7 users,
// even though PathRemoveFileSpecW is deprecated, it should still work.
// #include <pathcch.h>
// #pragma comment(lib, "pathcch.lib")
#endif

#include "../core/makedir.h"

using namespace std;

#ifdef OS_IS_WINDOWS

//On Windows, this function returns the executable's directory
vector<string> HomeData::getDefaultFilesDirs() {
  //HACK: add 2048 to the buffer size to be resilient to longer paths, beyond MAX_PATH.
  constexpr size_t bufSize = MAX_PATH + 2048;
  wchar_t buf[bufSize];
  DWORD length = GetModuleFileNameW(NULL, buf, bufSize);
  if(length <= 0) //failure
    throw StringError("Could not find containing directory of KataGo executable");
  if(length >= bufSize) //failure, path truncated
    throw StringError("Could not get containing directory of KataGo executable, path is too long");
  // #if (NTDDI_VERSION >= NTDDI_WIN8)
  // PathCchRemoveFileSpec(buf, bufSize);
  // #else
  PathRemoveFileSpecW(buf);
  // #endif
  constexpr size_t buf2Size = (bufSize+1) * 2;
  char buf2[buf2Size];
  size_t ret;
  wcstombs_s(&ret, buf2, buf2Size, buf, buf2Size-1);

  string executableDir(buf2);
  vector<string> dirs;
  dirs.push_back(executableDir);
  return dirs;
}

string HomeData::getDefaultFilesDirForHelpMessage() {
  return "(dir containing katago.exe)";
}


//On Windows, instead of home directory, we just make something inside the directory containing the executable
string HomeData::getHomeDataDir(bool makeDir, const string& homeDataDirOverride) {
  if(homeDataDirOverride != "") {
    if(makeDir) MakeDir::make(homeDataDirOverride);
    return homeDataDirOverride;
  }

  //HACK: add 2048 to the buffer size to be resilient to longer paths, beyond MAX_PATH.
  constexpr size_t bufSize = MAX_PATH + 2048;
  wchar_t buf[bufSize];
  DWORD length = GetModuleFileNameW(NULL, buf, bufSize);
  if(length <= 0) //failure
    throw StringError("Could not access containing directory of KataGo executable");
  if(length >= bufSize) //failure, path truncated
    throw StringError("Could not get containing directory of KataGo executable, path is too long");
  // #if (NTDDI_VERSION >= NTDDI_WIN8)
  // PathCchRemoveFileSpec(buf, bufSize);
  // #else
  PathRemoveFileSpecW(buf);
  // #endif
  constexpr size_t buf2Size = (bufSize+1) * 2;
  char buf2[buf2Size];
  size_t ret;
  wcstombs_s(&ret, buf2, buf2Size, buf, buf2Size-1);

  string homeDataDir(buf2);
  homeDataDir += "/KataGoData";
  if(makeDir) MakeDir::make(homeDataDir);
  return homeDataDir;
}
#endif

#ifdef OS_IS_UNIX_OR_APPLE
//On Linux, this function returns two locations:
//The directory containing the excutable.
//A katago-specific subdirectory of the home directory, same as getHomeDataDir.
vector<string> HomeData::getDefaultFilesDirs() {
  namespace gfs = ghc::filesystem;
  constexpr int bufSize = 2048;
  char result[bufSize];
  ssize_t count = readlink("/proc/self/exe", result, bufSize);
  vector<string> ret;
  if(count >= 0 && count < bufSize-1) {
    string exePath(result,count);
    const gfs::path path(exePath);
    string exeDir = path.parent_path().string();
    ret.push_back(exeDir);
  }
  ret.push_back(getHomeDataDir(false,""));
  return ret;
}

string HomeData::getDefaultFilesDirForHelpMessage() {
  return "(dir containing katago.exe, or else ~/.katago)";
}

string HomeData::getHomeDataDir(bool makeDir, const string& homeDataDirOverride) {
  if(homeDataDirOverride != "") {
    if(makeDir) MakeDir::make(homeDataDirOverride);
    return homeDataDirOverride;
  }

  string homeDataDir;
  const char* home =  getenv("HOME");
  if(home != NULL) {
    homeDataDir = string(home) + "/.katago";
    if(makeDir) MakeDir::make(homeDataDir);
    return homeDataDir;
  }

  int64_t bufSize = sysconf(_SC_GETPW_R_SIZE_MAX);
  if(bufSize == -1)
    bufSize = 1 << 16;
  char* buf = new char[bufSize];
  struct passwd pwd;
  struct passwd *result;
  int err = getpwuid_r(getuid(), &pwd, buf, bufSize, &result);
  if(err != 0) {
    delete[] buf;
    throw StringError("Could not find home directory for reading/writing data, errno " + Global::intToString(err));
  }
  //Just make something in the current directory
  if(result == NULL) {
    delete[] buf;
    homeDataDir = "./.katago";
    if(makeDir) MakeDir::make(homeDataDir);
    return homeDataDir;
  }
  homeDataDir = string(result->pw_dir);
  delete[] buf;
  homeDataDir += "/.katago";
  if(makeDir) MakeDir::make(homeDataDir);
  return homeDataDir;
}
#endif
