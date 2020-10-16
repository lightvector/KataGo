#ifdef BUILD_DISTRIBUTED

#ifndef DISTRIBUTED_HTTPLIB_WRAPPER_H_
#define DISTRIBUTED_HTTPLIB_WRAPPER_H_

//The point of this wrapper is to:
//1. Ensure CPPHTTPLIB_OPENSSL_SUPPORT and some other things are defined.
//2. Suppress a whole ton of warnings that you get when compiling with this header, by telling GCC to treat it like a system header.

#define CPPHTTPLIB_OPENSSL_SUPPORT
#define CPPHTTPLIB_ZLIB_SUPPORT
#pragma GCC system_header
#include <httplib.h>

#endif //HTTPLIB_WRAPPER_H_

#endif //BUILD_DISTRIBUTED
