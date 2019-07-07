#include "../dataio/homedata.h"
#include "../core/os.h"

#ifdef OS_IS_UNIX_OR_APPLE
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#endif

#include "../core/makedir.h"

using namespace std;

#ifdef OS_IS_WINDOWS
string HomeData::getHomeDataDir(bool makeDir) {
  //Just make something inside current directory
  string homeDataDir = "./KataGoData";
  if(makeDir) MakeDir::make(homeDataDir);
  return homeDataDir;
}
#endif

#ifdef OS_IS_UNIX_OR_APPLE
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
    throw StringError("Could not find home directory for reading/writing tuner or other data, errno " + Global::intToString(err));
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
