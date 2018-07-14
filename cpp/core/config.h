/*
 * Global defines and settings
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#define ONLYINDEBUG(x) x
//#define ONLYINDEBUG(x) ((void)0);

//#define NDEBUG  //this disables cassert
#include <cassert>

//Enable multithreading or disable it and replace threading stuff with single-threaded versions (see multithreading.h)
#define MULTITHREADING

#endif
