#include "../core/makedir.h"
#include "../core/os.h"

#ifdef OS_IS_WINDOWS
  #include <windows.h>
#endif
#ifdef OS_IS_UNIX_OR_APPLE
  #include <sys/types.h>
  #include <sys/stat.h>
#endif

#include <cerrno>
#include <ghc/filesystem.hpp>

namespace gfs = ghc::filesystem;

//------------------------
#include "../core/using.h"
//------------------------

//WINDOWS IMPLMENTATIION-------------------------------------------------------------

#ifdef OS_IS_WINDOWS

void MakeDir::make(const string& path) {
  gfs::path gfsPath(gfs::u8path(path));
  std::error_code ec;
  bool suc = gfs::create_directory(gfsPath, ec);
  if(!suc && ec) {
    if(ec == std::errc::file_exists)
      return;
    throw StringError("Error creating directory: " + ec.message());
  }
  //CreateDirectory(path.c_str(),NULL);
}

#endif

//UNIX IMPLEMENTATION------------------------------------------------------------------

#ifdef OS_IS_UNIX_OR_APPLE

void MakeDir::make(const string& path) {
  int result = mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  if(result != 0) {
    if(errno == EEXIST)
      return;
    throw StringError("Error creating directory: " + path);
  }
}

#endif
