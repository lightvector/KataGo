#include "../core/mainargs.h"

#include "../core/os.h"

#ifdef OS_IS_WINDOWS
#include <codecvt>
#include <windows.h>
#include <processenv.h>
#include <shellapi.h>
#endif

std::vector<std::string> MainArgs::getCommandLineArgsUTF8(int argc, const char* const* argv) {
#ifdef OS_IS_WINDOWS
  // Ignore argc and argv entirely and just call Windows-specific functions to get the full command line without
  // losing information in the case of non-ascii input.
  // Then convert to UTF8
  LPWSTR commandLine = GetCommandLineW();
  LPWSTR* argvWide = CommandLineToArgvW(commandLine,&argc);
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  std::vector<std::string> args;
  args.reserve(argc);
  for(int i = 0; i<argc; i++)
    args.push_back(converter.to_bytes(argvWide[i]));
  return args;
#else
  // For non-Windows, for now assume we have UTF8. If we need to add a case for another OS here, we can do that later.
  std::vector<std::string> args;
  args.reserve(argc);
  for(int i = 0; i<argc; i++)
    args.push_back(std::string(argv[i]));
  return args;
#endif
}
