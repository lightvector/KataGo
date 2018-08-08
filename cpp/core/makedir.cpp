#include "../core/makedir.h"

#ifdef _WIN32
 #define _IS_WINDOWS
#elif _WIN64
 #define _IS_WINDOWS
#elif __unix || __APPLE__
  #define _IS_UNIX
#else
 #error Unknown OS!
#endif

#ifdef _IS_WINDOWS
  #include <windows.h>
#endif
#ifdef _IS_UNIX
  #include <sys/types.h>
  #include <sys/stat.h>
#endif

//WINDOWS IMPLMENTATIION-------------------------------------------------------------

#ifdef _IS_WINDOWS

void MakeDir::make(const string& path) {
  CreateDirectory(path.c_str(),NULL);
}

#endif

//UNIX IMPLEMENTATION------------------------------------------------------------------

#ifdef _IS_UNIX

void MakeDir::make(const string& path) {
  mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

#endif
