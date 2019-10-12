#ifndef CORE_BSEARCH_H_
#define CORE_BSEARCH_H_

#include <cstring>
#include <stdint.h>

namespace BSearch {

  //Assumes arr is sorted.
  //Finds the first index i within [low,high) where arr[i] > x, or high if such an index does not exist.
  size_t findFirstGt(const double* arr, double x, size_t low, size_t high);

  //TESTING----------------------------------------------
  void runTests();
}

#endif //CORE_BSEARCH_H_
