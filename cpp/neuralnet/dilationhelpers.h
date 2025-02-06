#ifndef DILATION_HELPERS_H
#define DILATION_HELPERS_H

namespace DilationHelpers {

  inline int divRoundUp(int x, int n) {
    return (x + n-1) / n;
  }
  inline int padToMultiple(int x, int n) {
    return divRoundUp(x,n) * n - x;
  }

}


#endif
