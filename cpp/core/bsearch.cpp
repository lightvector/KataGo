#include "../core/bsearch.h"

#include "../core/test.h"

size_t BSearch::findFirstGt(const double* arr, double x, size_t low, size_t high) {
  if(low >= high)
    return high;
  size_t mid = (low+high)/2;
  if(arr[mid] > x)
    return findFirstGt(arr,x,low,mid);
  else
    return findFirstGt(arr,x,mid+1,high);
}


void BSearch::runTests() {
  constexpr size_t len = 13;
  double arr[len] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  for(size_t i = 0; i<len; i++) {
    testAssert(findFirstGt(arr,(double)i,0,len) == i+1);
    testAssert(findFirstGt(arr,(double)i+0.7,0,len) == i+1);
    testAssert(findFirstGt(arr,(double)i+0.99,0,len) == i+1);
    testAssert(findFirstGt(arr,(double)i-0.01,0,len) == i);
  }
}
