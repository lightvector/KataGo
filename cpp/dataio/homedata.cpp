#include "../dataio/homedata.h"
#include "../core/os.h"

#ifdef OS_IS_UNIX_OR_APPLE
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#endif
#ifdef OS_IS_WINDOWS
#include <Windows.h>
#include <Shlwapi.h>
#endif

#include "../core/makedir.h"

using namespace std;

#ifdef OS_IS_WINDOWS

//On Windows, this function returns the executable's directory
string HomeData::getDefaultFilesDir() {
  size_t bufSize = MAX_PATH;
  TCHAR buf[bufSize];
  DWORD length = GetModuleFileName(NULL, buf, bufSize);
  if(length <= 0) //failure
    throw StringError("Could not find containing directory of KataGo executable");
  #if (NTDDI_VERSION >= NTDDI_WIN8)
  PathCchRemoveFileSpec(buf, bufSize);
  #else
  PathRemoveFileSpec(buf);
  #endif
  string executableDir(buf);
  return executableDir;
}

string HomeData::getDefaultFilesDirForHelpMessage() {
  return "(dir containing katago.exe)"
}


//On Windows, instead of home directory, we just make something inside the directory containing the executable
string HomeData::getHomeDataDir(bool makeDir) {
  size_t bufSize = MAX_PATH;
  TCHAR buf[bufSize];
  DWORD length = GetModuleFileName(NULL, buf, bufSize);
  if(length <= 0) //failure
    throw StringError("Could not access containing directory of KataGo executable");
  #if (NTDDI_VERSION >= NTDDI_WIN8)
  PathCchRemoveFileSpec(buf, bufSize);
  #else
  PathRemoveFileSpec(buf);
  #endif

  string homeDataDir(buf);
  homeDataDir += "/KataGoData";
  if(makeDir) MakeDir::make(homeDataDir);
  return homeDataDir;
}
#endif

#ifdef OS_IS_UNIX_OR_APPLE
//On Linux, this function returns a katago-specific subdirectory of the home directory, same as getHomeDataDir
string HomeData::getDefaultFilesDir() {
  return getHomeDataDir(false);
}

string HomeData::getDefaultFilesDirForHelpMessage() {
  return "~/.katago";
}

string HomeData::getHomeDataDir(bool makeDir) {
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
