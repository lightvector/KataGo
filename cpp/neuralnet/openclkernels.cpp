#include "../neuralnet/openclkernels.h"

using namespace std;

//TODO these are all fairly naive implementations just to get things working
//optimize them or find a library with more optimized implementations

string OpenCLKernels::conv2dNCHW = R"%%(

//Spatial size of tile loaded into local memory, not counting filterRadius
#ifndef TILE_XSIZE
#define TILE_XSIZE 32
#endif
#ifndef TILE_YSIZE
#define TILE_YSIZE 4
#endif

//Channel depth of tile loaded into local memory
#ifndef TILE_CHANNELS
#define TILE_CHANNELS 4
#endif

//group id 0 indexes different tiles along x dimension
//group id 1 indexes different tiles along y dimension
//local id 0 indexes threads that parallelize xwise across the internal of a tile, must be a factor of TILE_XSIZE
//local id 1 indexes threads that parallelize ywise across the internal of a tile, must be a factor of TILE_YSIZE
//group id 2 indexes different output channels
//local id 2 is ASSUMED to be always 0, with local size 2 ASSUMED to be 1, so that we don't need to index these local memory space on this dimension
__kernel void conv2dNCHW(
  __global float* restrict input,  //N, ic, H, W
  __global float* restrict filter, //oc, ic, fy, fx
  __global float* restrict output, //N, oc, H, W

  __local float* restrict inputTile, //ic, H, W      size = TILE_CHANNELS * inputTileXSize * inputTileYSize
  __local float* restrict outputTile, //H, W         size = TILE_XSIZE * TILE_YSIZE

  int nSize,
  int xSize,
  int ySize,
  int ocSize,
  int icSize,
  int filterXRadius,
  int filterYRadius
) {
  const int xBase = get_group_id(0) * TILE_XSIZE;
  const int yBase = get_group_id(1) * TILE_YSIZE;
  const int oc = get_global_id(2);

  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int lxSize = get_local_size(0);
  const int lySize = get_local_size(1);

  //The input tile is filterXRadius*2 or filterYRadius*2 larger than the tile size
  const int inputTileXSize = TILE_XSIZE + filterXRadius * 2;
  const int inputTileYSize = TILE_YSIZE + filterYRadius * 2;

  const int xySize = xSize * ySize;

  const int fxSize = (2 * filterXRadius + 1);
  const int fySize = (2 * filterYRadius + 1);

#define INPUT(_n,_ic,_y,_x) input[((_n) * icSize + (_ic)) * xySize + (_y) * xSize + (_x)]
#define INPUTTILE(_ic,_ity,_itx) inputTile[((_ic) * inputTileYSize + (_ity)) * inputTileXSize + (_itx)]

#define FILTER(_oc,_ic,_y,_x) filter[(((_oc) * icSize + (_ic)) * fySize + (_y)) * fxSize + (_x)]

#define OUTPUT(_n,_oc,_y,_x) output[((_n) * ocSize + (_oc)) * xySize + (_y) * xSize + (_x)]
#define OUTPUTTILE(_oty,_otx) outputTile[(_oty) * TILE_XSIZE + (_otx)]

  for(int n = 0; n < nSize; n++) {
    float acc = 0.0f;

    //Initialize outputTile. No need to sync for this tile since each thread only ever reads its own spots
    for(int oty = ly; oty<TILE_YSIZE; oty += lySize) {
      for(int otx = lx; otx<TILE_XSIZE; otx += lxSize) {
        OUTPUTTILE(oty,otx) = 0.0f;
      }
    }

    //Walk over chunks of TILE_CHANNELS many input channels at a time
    for(int icBase = 0; icBase<icSize; icBase += TILE_CHANNELS) {

      //Copy input tile using local threads in parallel
      for(int dic = 0; dic<TILE_CHANNELS && icBase+dic < icSize; dic += 1) {
        for(int ity = ly; ity<inputTileYSize; ity += lySize) {
          int iy = ity+yBase-filterYRadius;
          for(int itx = lx; itx<inputTileXSize; itx += lxSize) {
            int ix = itx+xBase-filterXRadius;
            float inputValue = 0.0f;
            if(iy >= 0 && iy < ySize && ix >= 0 && ix < xSize) {
              inputValue = INPUT(n,icBase+dic,iy,ix);
            }
            INPUTTILE(dic,ity,itx) = inputValue;
          }
        }
      }

      //Synchronize!
      barrier(CLK_LOCAL_MEM_FENCE);

      //Accumulate this convolution block into output tile.
      //Iterate over the bits in this tile that the thread is responsible for
      for(int oty = ly; oty<TILE_YSIZE; oty += lySize) {
        for(int otx = lx; otx<TILE_XSIZE; otx += lxSize) {

          //And then perform the convolution to accumulate that bit
          float acc = 0.0f;
          for(int dic = 0; dic<TILE_CHANNELS && icBase+dic < icSize; dic += 1) {
            for(int fy = 0; fy < fySize; fy++) {
              for(int fx = 0; fx < fxSize; fx++) {
                acc += INPUTTILE(dic,oty+fy,otx+fx) * FILTER(oc,icBase+dic,fy,fx);
              }
            }
          }
          OUTPUTTILE(oty,otx) += acc;
        }
      }
    } //close loop over input channel chunks

    //Now, write tile contents back into output
    for(int oty = ly; oty<TILE_YSIZE; oty += lySize) {
      int oy = yBase+oty;
      for(int otx = lx; otx<TILE_XSIZE; otx += lxSize) {
        int ox = xBase+otx;
        if(oy >= 0 && oy < ySize && ox >= 0 && ox < xSize) {
          OUTPUT(n, oc, yBase+oty, xBase+otx) = OUTPUTTILE(oty,otx);
        }
      }
    }
  } //Close loop over batch
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
  const int oc = get_global_id(0);
  const int n = get_global_id(1);

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

string OpenCLKernels::valueHeadPoolChannelsNCHW = R"%%(
__kernel void valueHeadPoolChannelsNCHW(
  __global float* input,  //N, c, HW
  __global float* output, //N, c
  __global float* maskSums, //N
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

    float div = maskSums[n];
    float sqrtdiv = sqrt(div);
    float finalMean = finalSum/div;

    int outBase = n * cSize * 3 + c;
    output[outBase] = finalMean;
    output[outBase + cSize] = finalMean * (sqrtdiv - 14.0f) * 0.1f;
    output[outBase + cSize*2] = finalMean * ((sqrtdiv - 14.0f) * (sqrtdiv - 14.0f) * 0.01f - 0.1f);
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


string OpenCLKernels::addCBiasesNC = R"%%(
__kernel void addCBiasesNC(
  __global float* accum,  //N,C
  __global float* biases, //C
  int nSize,
  int cSize
) {
  const int c = get_global_id(0);
  const int n = get_global_id(1);

  if(n < nSize && c < cSize)
    accum[n * cSize + c] += biases[c];
}
)%%";


string OpenCLKernels::addCBiasesNCRelu = R"%%(
__kernel void addCBiasesNCRelu(
  __global float* accum,  //N,C
  __global float* biases, //C
  int nSize,
  int cSize
) {
  const int c = get_global_id(0);
  const int n = get_global_id(1);

  if(n < nSize && c < cSize)
    accum[n * cSize + c] = fmax(accum[n * cSize + c] + biases[c], 0.0f);
}
)%%";


string OpenCLKernels::transposeNCHW = R"%%(
__kernel void transposeNCHW(
  __global float* in,
  __global float* out,
  //+1 avoids bank conflicts
  __local float* tileNCHW, //size = tileDim * (tileDim+1) * get_local_size(2)
  int xSize,
  int ySize,
  int tileDim,
  int tileStride,
  int ncSize
) {
  const int tileDimP1 = tileDim+1;
  const int xIdx = get_global_id(0);
  const int yIdx = get_global_id(1);
  const int xLocal = get_local_id(0);
  const int yLocal = get_local_id(1);
  const int nc = get_global_id(2);
  const int xySize = xSize * ySize;

  if(xIdx < xSize && nc < ncSize) {
    for(int j = 0; j < tileDim && yIdx+j < ySize; j += tileStride) {
      int inIdx = xIdx + xSize * (yIdx+j) + xySize * nc;
      tileNCHW[(yLocal+j)*tileDimP1 + xLocal] = in[inIdx];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  //Transpose idx
  int outXIdx = get_group_id(1) * tileDim + xLocal;
  int outYIdx = get_group_id(0) * tileDim + yLocal;

  if(outXIdx < ySize && nc < ncSize) {
    for(int j = 0; j < tileDim && outYIdx+j < xSize; j += tileStride) {
      int outIdx = outXIdx + ySize * (outYIdx+j) + xySize * nc;
      out[outIdx] = tileNCHW[xLocal*tileDimP1 + yLocal+j];
    }
  }
}
)%%";


string OpenCLKernels::mirror = R"%%(
__kernel void mirror(__global float* in, __global float* out, int batchSize, int mSize, int subSize)
{
  const int subIdx = get_global_id(0);
  const int mIdx = get_global_id(1);
  const int batchIdx = get_global_id(2);
  if(subIdx < subSize && mIdx < mSize && batchIdx < batchSize) {
    int inIdx = subIdx + subSize * (mIdx + mSize * batchIdx);
    int outIdx = subIdx + subSize * ((mSize-mIdx-1) + mSize * batchIdx);
    out[outIdx] = in[inIdx];
  }
}
)%%";


string OpenCLKernels::extractChannel0NCHW = R"%%(
__kernel void extractChannel0NCHW(__global float* in, __global float* out, int nSize, int cSize, int xySize)
{
  const int xyIdx = get_global_id(0);
  const int nIdx = get_global_id(1);
  if(xyIdx < xySize && nIdx < nSize) {
    out[nIdx * xySize + xyIdx] = in[nIdx * cSize * xySize + xyIdx];
  }
}
)%%";
