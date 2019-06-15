#include "../neuralnet/openclkernels.h"

using namespace std;

string OpenCLKernels::conv2dNCHW = R"%%(
__kernel void conv2dNCHW(
  __global float* input,  //N, ic, H, W
  __global float* filter, //oc, ic, fy, fx
  __global float* output, //N, oc, H, W
  int nSize,
  int ySize,
  int xSize,
  int ocSize,
  int icSize,
  int filterYRadius,
  int filterXRadius
){
  const int ox = get_global_id(0);
  const int oy = get_global_id(1);
  const int oc = get_global_id(2);

  const int yxSize = ySize * xSize;
  const int ocyxSize = ocSize * yxSize;
  const int icyxSize = icSize * yxSize;

  const int fxSize = (2 * filterXRadius + 1);
  const int fyxSize = (2 * filterYRadius + 1) * fxSize;
  const int ficyxSize = icSize * fyxSize;

  if(oc < ocSize && ox < xSize && oy < ySize) {
    for(int n = 0; n < nSize; n++) {
      float acc = 0.0f;
      for(int ic = 0; ic < icSize; ic++) {
        for(int dy = -filterYRadius; dy <= filterYRadius; dy++) {
          int y = oy + dy;
          int fy = dy + filterYRadius;
          if(y >= 0 && y < ySize) {
            for(int dx = -filterXRadius; dx <= filterXRadius; dx++) {
              int x = ox + dx;
              int fx = dx + filterXRadius;
              if(x >= 0 && x < xSize) {
                acc +=
                  input[n * icyxSize + ic * yxSize + y * xSize + x]
                  * filter[oc * ficyxSize + ic * fyxSize + fy * fxSize + fx];
              }
            }
          }
        }
      }

      output[n * ocyxSize + oc * yxSize + oy * xSize + ox] = acc;
    }
  }
}

)%%";
