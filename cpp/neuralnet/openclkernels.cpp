#include "../neuralnet/openclkernels.h"

using namespace std;

//TODO these are all fairly naive implementations just to get things working
//optimize them or find a library with more optimized implementations

string OpenCLKernels::conv2dNCHW = R"%%(
__kernel void conv2dNCHW(
  __global float* input,  //N, ic, H, W
  __global float* filter, //oc, ic, fy, fx
  __global float* output, //N, oc, H, W
  int nSize,
  int xSize,
  int ySize,
  int ocSize,
  int icSize,
  int filterXRadius,
  int filterYRadius
) {
  const int ox = get_global_id(0);
  const int oy = get_global_id(1);
  const int oc = get_global_id(2);

  const int xySize = xSize * ySize;
  const int ocxySize = ocSize * xySize;
  const int icxySize = icSize * xySize;

  const int fxSize = (2 * filterXRadius + 1);
  const int fxySize = (2 * filterYRadius + 1) * fxSize;
  const int ficxySize = icSize * fxySize;

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
                  input[n * icxySize + ic * xySize + y * xSize + x]
                  * filter[oc * ficxySize + ic * fxySize + fy * fxSize + fx];
              }
            }
          }
        }
      }

      output[n * ocxySize + oc * xySize + oy * xSize + ox] = acc;
    }
  }
}

)%%";

string OpenCLKernels::scaleBiasMaskNCHW = R"%%(
__kernel void scaleBiasMaskNCHW(
  __global float* input,  //N, c, H, W
  __global float* output, //N, c, H, W
  __global float* scale,  //c
  __global float* bias,   //c
  __global float* mask,   //N, H, W
  int nSize,
  int cSize,
  int xySize
) {
  const int xy = get_global_id(0);
  const int c = get_global_id(1);

  if(c < cSize && xy < xySize) {
    for(int n = 0; n < nSize; n++) {
      int idx = (n * cSize + c) * xySize + xy;
      output[idx] = (input[idx] * scale[c] + bias[c]) * mask[n * xySize + xy];
    }
  }
}
)%%";

string OpenCLKernels::scaleBiasMaskReluNCHW = R"%%(
__kernel void scaleBiasMaskReluNCHW(
  __global float* input,  //N, c, H, W
  __global float* output, //N, c, H, W
  __global float* scale,  //c
  __global float* bias,   //c
  __global float* mask,   //N, H, W
  int nSize,
  int cSize,
  int xySize
) {
  const int xy = get_global_id(0);
  const int c = get_global_id(1);

  if(c < cSize && xy < xySize) {
    for(int n = 0; n < nSize; n++) {
      int idx = (n * cSize + c) * xySize + xy;
      output[idx] = fmax(input[idx] * scale[c] + bias[c], 0.0f) * mask[n * xySize + xy];
    }
  }
}
)%%";

string OpenCLKernels::addPointWise = R"%%(
__kernel void addPointWise(
  __global float* accum,
  __global float* value,
  int size
) {
  const int s = get_global_id(0);

  if(s < size)
    accum[s] += value[s];
}
)%%";


string OpenCLKernels::matMul = R"%%(
__kernel void matMul(
  __global float* input,  //N, ic
  __global float* weights, //oc, ic
  __global float* output,  //N, oc
  int nSize,
  int icSize,
  int ocSize
) {
  const int n = get_global_id(0);
  const int oc = get_global_id(1);

  if(n < nSize && oc < ocSize) {
    float acc = 0.0f;
    for(int ic = 0; ic < icSize; ic++)
      acc += input[n * icSize + ic] * weights[oc * icSize + ic];
    output[n * ocSize + oc] = acc;
  }
}
)%%";

string OpenCLKernels::sumChannelsNCHW = R"%%(
__kernel void sumChannelsNCHW(
  __global float* input,  //N, c, HW
  __global float* output, //N, c
  __local float* partialSums, //size = get_local_size(0) * get_local_size(1) * get_local_size(2)
  int nSize,
  int cSize,
  int xySize
) {
  //PRECONDIION: Kernel is being run where get_num_groups(0) == 1, so that global id and local id are identical for dim 0
  const int xyBase = get_local_id(0);
  const int xyStride = get_local_size(0);
  const int c = get_global_id(1);
  const int n = get_global_id(2);
  const int localId1 = get_local_id(1);
  const int localSize1 = get_local_size(1);
  const int localId2 = get_local_id(2);

  float sum = 0.0f;
  if(n < nSize && c < cSize) {
    //Sum up the elements that this group member is responsible for
    for(int xy = xyBase; xy < xySize; xy += xyStride) {
      int idx = (n * cSize + c) * xySize + xy;
      float v = input[idx];
      sum += v;
    }
  }

  //Write to local memory for performing the reduction
  int localIdx = (localId2 * localSize1 + localId1) * xyStride + xyBase;
  partialSums[localIdx] = sum;

  //Parallel folding downward
  for(int span = xyStride / 2; span > 0; span /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);

    if(xyBase < span) {
      partialSums[localIdx] += partialSums[localIdx + span];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if(n < nSize && c < cSize && xyBase == 0) {
    float finalSum = partialSums[localIdx];
    int outBase = n * cSize + c;
    output[outBase] = finalSum;
  }
}
)%%";


string OpenCLKernels::gPoolChannelsNCHW = R"%%(
__kernel void gPoolChannelsNCHW(
  __global float* input,  //N, c, HW
  __global float* output, //N, c
  __global float* maskSums, //N
  __local float* partialSums, //size = get_local_size(0) * get_local_size(1) * get_local_size(2)
  __local float* partialMaxes, //size = get_local_size(0) * get_local_size(1) * get_local_size(2)
  int nSize,
  int cSize,
  int xySize
) {
  //PRECONDIION: Kernel is being run where get_num_groups(0) == 1, so that global id and local id are identical for dim 0
  const int xyBase = get_local_id(0);
  const int xyStride = get_local_size(0);
  const int c = get_global_id(1);
  const int n = get_global_id(2);
  const int localId1 = get_local_id(1);
  const int localSize1 = get_local_size(1);
  const int localId2 = get_local_id(2);

  float sum = 0.0f;
  float max = 0.0f;
  if(n < nSize && c < cSize) {
    //Sum up the elements that this group member is responsible for
    for(int xy = xyBase; xy < xySize; xy += xyStride) {
      int idx = (n * cSize + c) * xySize + xy;
      float v = input[idx];
      sum += v;
      max = fmax(max,v);
    }
  }

  //Write to local memory for performing the reduction
  int localIdx = (localId2 * localSize1 + localId1) * xyStride + xyBase;
  partialSums[localIdx] = sum;
  partialMaxes[localIdx] = max;

  //Parallel folding downward
  for(int span = xyStride / 2; span > 0; span /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);

    if(xyBase < span) {
      partialSums[localIdx] += partialSums[localIdx + span];
      partialMaxes[localIdx] = fmax(partialMaxes[localIdx], partialMaxes[localIdx + span]);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if(n < nSize && c < cSize && xyBase == 0) {
    float finalSum = partialSums[localIdx];
    float finalMax = partialMaxes[localIdx];

    float div = maskSums[n];
    float sqrtdiv = sqrt(div);
    float finalMean = finalSum/div;

    int outBase = n * cSize * 3 + c;
    output[outBase] = finalMean;
    output[outBase + cSize] = finalMean * (sqrtdiv - 14.0f) * 0.1f;
    output[outBase + cSize*2] = finalMax;
  }
}
)%%";


string OpenCLKernels::addChannelBiasesNCHW = R"%%(
__kernel void addChannelBiasesNCHW(
  __global float* accum,  //NC, HW
  __global float* biases, //NC
  int ncSize,
  int xySize
) {
  const int xy = get_global_id(0);
  const int nc = get_global_id(1);

  if(nc < ncSize && xy < xySize)
    accum[nc * xySize + xy] += biases[nc];
}
)%%";
