/*
 * Global defines and settings
 */

#ifndef CORE_CONFIG_H_
#define CORE_CONFIG_H_

#define ONLYINDEBUG(x) x
//#define ONLYINDEBUG(x) ((void)0);

//#define NDEBUG  //this disables cassert
#include <cassert>

//Enable multithreading or disable it and replace threading stuff with single-threaded versions (see multithreading.h)
#define MULTITHREADING

#endif  // CORE_CONFIG_H_
