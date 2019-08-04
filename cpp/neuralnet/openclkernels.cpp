#ifdef USE_OPENCL_BACKEND

#include "../neuralnet/openclkernels.h"

using namespace std;

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


string OpenCLKernels::winogradConvNCHW = R"%%(

//Expected defines---------------------------------

//Dimension of input tile
//INTILE_XSIZE 4 for F(2x2,3x3)
//INTILE_YSIZE 4 for F(2x2,3x3)

//Dimension of conv
//CONV_XSIZE 3 for F(2x2,3x3)
//CONV_YSIZE 3 for F(2x2,3x3)

//Output tile size
//OUTTILE_XSIZE 2 for F(2x2,3x3)
//OUTTILE_YSIZE 2 for F(2x2,3x3)

//Location of the upper left corner of the zeroth tile
//INTILE_XOFFSET (-1) for F(2x2,3x3)
//INTILE_YOFFSET (-1) for F(2x2,3x3)

#define SQRT8 2.82842712475f
#define SQRT2 1.41421356237f
#define SQRTHALF 0.70710678118f
#define SQRTEIGHTH 0.35355339059f

__kernel void transform(
  __global float* restrict input,  //N, ic, H, W
  __global float* restrict transformed, //INTILE_YSIZE, INTILE_XSIZE, ic, batch, tileY, tileX
  int nSize,
  int xSize,
  int ySize,
  int numTilesX,
  int numTilesY,
  int icSize
) {
  int id0 = get_global_id(0);
  const int ntxty = id0;
  const int tileX = id0 % numTilesX;
  id0 = id0 / numTilesX;
  const int tileY = id0 % numTilesY;
  id0 = id0 / numTilesY;
  const int n = id0;
  const int ic = get_global_id(1);
  const int nic = n * icSize + ic;
  const int xySize = xSize * ySize;

#define INPUT(_nic,_xy) input[((_nic) * xySize) + (_xy)]
#define WTILE(_y,_x) wTile[(_y)*INTILE_XSIZE + (_x)]

  __private float wTile[INTILE_XSIZE * INTILE_YSIZE];

  //Copy input into private tile
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    int y = tileY * OUTTILE_YSIZE + subY + INTILE_YOFFSET;
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      int x = tileX * OUTTILE_XSIZE + subX + INTILE_XOFFSET;
      float value = 0.0f;
      if(y >= 0 && y < ySize && x >= 0 && x < xSize && tileX < numTilesX && tileY < numTilesY && n < nSize && ic < icSize) {
        int xy = y * xSize + x;
        value = INPUT(nic,xy);
      }
      WTILE(subY,subX) = value;
    }
  }

#if CONV_XSIZE == 3 && OUTTILE_XSIZE == 2
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    float z0 = WTILE(subY,0);
    float z1 = WTILE(subY,1);
    float z2 = WTILE(subY,2);
    float z3 = WTILE(subY,3);
    WTILE(subY,0) = z0 - z2;
    WTILE(subY,1) = z1 + z2;
    WTILE(subY,2) = z2 - z1;
    WTILE(subY,3) = z1 - z3;
  }
#elif CONV_XSIZE == 3 && OUTTILE_XSIZE == 4
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    float z0 = WTILE(subY,0);
    float z1 = WTILE(subY,1);
    float z2 = WTILE(subY,2);
    float z3 = WTILE(subY,3);
    float z4 = WTILE(subY,4);
    float z5 = WTILE(subY,5);
    // Low error winograd
    // WTILE(subY,0) = z0 - 2.5f*z2 + z4;
    // WTILE(subY,1) = - SQRT2*z1 - 2.0f*z2 + SQRTHALF*z3 + z4;
    // WTILE(subY,2) =   SQRT2*z1 - 2.0f*z2 - SQRTHALF*z3 + z4;
    // WTILE(subY,3) = - SQRTHALF*z1 - 0.5f*z2 + SQRT2*z3 + z4;
    // WTILE(subY,4) =   SQRTHALF*z1 - 0.5f*z2 - SQRT2*z3 + z4;
    // WTILE(subY,5) = z1 - 2.5f*z3 + z5;
    WTILE(subY,0) = 4.0f*z0 - 5.0f*z2 + z4;
    WTILE(subY,1) = - 4.0f*z1 - 4.0f*z2 + z3 + z4;
    WTILE(subY,2) =   4.0f*z1 - 4.0f*z2 - z3 + z4;
    WTILE(subY,3) = - 2.0f*z1 - z2 + 2.0f*z3 + z4;
    WTILE(subY,4) =   2.0f*z1 - z2 - 2.0f*z3 + z4;
    WTILE(subY,5) = 4.0f*z1 - 5.0f*z3 + z5;
  }
#elif CONV_XSIZE == 5 && OUTTILE_XSIZE == 2
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    float z0 = WTILE(subY,0);
    float z1 = WTILE(subY,1);
    float z2 = WTILE(subY,2);
    float z3 = WTILE(subY,3);
    float z4 = WTILE(subY,4);
    float z5 = WTILE(subY,5);
    WTILE(subY,0) = 4.0f*z0 - 5.0f*z2 + z4;
    WTILE(subY,1) = - 4.0f*z1 - 4.0f*z2 + z3 + z4;
    WTILE(subY,2) =   4.0f*z1 - 4.0f*z2 - z3 + z4;
    WTILE(subY,3) = - 2.0f*z1 - z2 + 2.0f*z3 + z4;
    WTILE(subY,4) =   2.0f*z1 - z2 - 2.0f*z3 + z4;
    WTILE(subY,5) = 4.0f*z1 - 5.0f*z3 + z5;
  }
#else
  #error "No X winograd implemented for this conv and tile size"
#endif

#if CONV_YSIZE == 3 && OUTTILE_YSIZE == 2
  for(int subX = 0; subX < INTILE_XSIZE; subX++) {
    float z0 = WTILE(0,subX);
    float z1 = WTILE(1,subX);
    float z2 = WTILE(2,subX);
    float z3 = WTILE(3,subX);
    WTILE(0,subX) = z0 - z2;
    WTILE(1,subX) = z1 + z2;
    WTILE(2,subX) = z2 - z1;
    WTILE(3,subX) = z1 - z3;
  }
#elif CONV_YSIZE == 3 && OUTTILE_YSIZE == 4
  for(int subX = 0; subX < INTILE_XSIZE; subX++) {
    float z0 = WTILE(0,subX);
    float z1 = WTILE(1,subX);
    float z2 = WTILE(2,subX);
    float z3 = WTILE(3,subX);
    float z4 = WTILE(4,subX);
    float z5 = WTILE(5,subX);
    // Low error winograd
    // WTILE(0,subX) = z0 - 2.5f*z2 + z4;
    // WTILE(1,subX) = - SQRT2*z1 - 2.0f*z2 + SQRTHALF*z3 + z4;
    // WTILE(2,subX) =   SQRT2*z1 - 2.0f*z2 - SQRTHALF*z3 + z4;
    // WTILE(3,subX) = - SQRTHALF*z1 - 0.5f*z2 + SQRT2*z3 + z4;
    // WTILE(4,subX) =   SQRTHALF*z1 - 0.5f*z2 - SQRT2*z3 + z4;
    // WTILE(5,subX) = z1 - 2.5f*z3 + z5;
    WTILE(0,subX) = 4.0f*z0 - 5.0f*z2 + z4;
    WTILE(1,subX) = - 4.0f*z1 - 4.0f*z2 + z3 + z4;
    WTILE(2,subX) =   4.0f*z1 - 4.0f*z2 - z3 + z4;
    WTILE(3,subX) = - 2.0f*z1 - z2 + 2.0f*z3 + z4;
    WTILE(4,subX) =   2.0f*z1 - z2 - 2.0f*z3 + z4;
    WTILE(5,subX) = 4.0f*z1 - 5.0f*z3 + z5;
  }
#elif CONV_YSIZE == 5 && OUTTILE_YSIZE == 2
  for(int subX = 0; subX < INTILE_XSIZE; subX++) {
    float z0 = WTILE(0,subX);
    float z1 = WTILE(1,subX);
    float z2 = WTILE(2,subX);
    float z3 = WTILE(3,subX);
    float z4 = WTILE(4,subX);
    float z5 = WTILE(5,subX);
    WTILE(0,subX) = 4.0f*z0 - 5.0f*z2 + z4;
    WTILE(1,subX) = - 4.0f*z1 - 4.0f*z2 + z3 + z4;
    WTILE(2,subX) =   4.0f*z1 - 4.0f*z2 - z3 + z4;
    WTILE(3,subX) = - 2.0f*z1 - z2 + 2.0f*z3 + z4;
    WTILE(4,subX) =   2.0f*z1 - z2 - 2.0f*z3 + z4;
    WTILE(5,subX) = 4.0f*z1 - 5.0f*z3 + z5;
  }
#else
  #error "No Y winograd implemented for this conv and tile size"
#endif

#define TRANS(_suby,_subx,_ic,_ntile) transformed[(((_suby) * INTILE_XSIZE + (_subx))*icSize + (_ic)) * ntxtySize + (_ntile)]

  if(tileX < numTilesX && tileY < numTilesY && n < nSize && ic < icSize) {
    const int ntxtySize = nSize * numTilesX * numTilesY;
    const int ntile = (n * numTilesY + tileY) * numTilesX + tileX;

    //Copy private tile out to transformed output
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      for(int subX = 0; subX < INTILE_XSIZE; subX++) {
        TRANS(subY,subX,ic,ntile) = WTILE(subY,subX);
      }
    }
  }

}

__kernel void bnReluTransform(
  __global float* restrict input,  //N, ic, H, W
  __global float* restrict transformed, //INTILE_YSIZE, INTILE_XSIZE, ic, batch, tileY, tileX
  __global float* restrict scale, //ic
  __global float* restrict bias, //ic
  __global float* restrict mask, //N, H, W
  int nSize,
  int xSize,
  int ySize,
  int numTilesX,
  int numTilesY,
  int icSize
) {
  int id0 = get_global_id(0);
  const int ntxty = id0;
  const int tileX = id0 % numTilesX;
  id0 = id0 / numTilesX;
  const int tileY = id0 % numTilesY;
  id0 = id0 / numTilesY;
  const int n = id0;
  const int ic = get_global_id(1);
  const int nic = n * icSize + ic;
  const int xySize = xSize * ySize;

#define INPUT(_nic,_xy) input[((_nic) * xySize) + (_xy)]
#define WTILE(_y,_x) wTile[(_y)*INTILE_XSIZE + (_x)]

  __private float wTile[INTILE_XSIZE * INTILE_YSIZE];

  //Copy input into private tile
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    int y = tileY * OUTTILE_YSIZE + subY + INTILE_YOFFSET;
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      int x = tileX * OUTTILE_XSIZE + subX + INTILE_XOFFSET;
      float value = 0.0f;
      if(y >= 0 && y < ySize && x >= 0 && x < xSize && tileX < numTilesX && tileY < numTilesY && n < nSize && ic < icSize) {
        int xy = y * xSize + x;
        value = fmax(INPUT(nic,xy) * scale[ic] + bias[ic], 0.0f) * mask[n * xySize + xy];
      }
      WTILE(subY,subX) = value;
    }
  }

#if CONV_XSIZE == 3 && OUTTILE_XSIZE == 2
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    float z0 = WTILE(subY,0);
    float z1 = WTILE(subY,1);
    float z2 = WTILE(subY,2);
    float z3 = WTILE(subY,3);
    WTILE(subY,0) = z0 - z2;
    WTILE(subY,1) = z1 + z2;
    WTILE(subY,2) = z2 - z1;
    WTILE(subY,3) = z1 - z3;
  }
#elif CONV_XSIZE == 3 && OUTTILE_XSIZE == 4
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    float z0 = WTILE(subY,0);
    float z1 = WTILE(subY,1);
    float z2 = WTILE(subY,2);
    float z3 = WTILE(subY,3);
    float z4 = WTILE(subY,4);
    float z5 = WTILE(subY,5);
    // Low error winograd
    // WTILE(subY,0) = z0 - 2.5f*z2 + z4;
    // WTILE(subY,1) = - SQRT2*z1 - 2.0f*z2 + SQRTHALF*z3 + z4;
    // WTILE(subY,2) =   SQRT2*z1 - 2.0f*z2 - SQRTHALF*z3 + z4;
    // WTILE(subY,3) = - SQRTHALF*z1 - 0.5f*z2 + SQRT2*z3 + z4;
    // WTILE(subY,4) =   SQRTHALF*z1 - 0.5f*z2 - SQRT2*z3 + z4;
    // WTILE(subY,5) = z1 - 2.5f*z3 + z5;
    WTILE(subY,0) = 4.0f*z0 - 5.0f*z2 + z4;
    WTILE(subY,1) = - 4.0f*z1 - 4.0f*z2 + z3 + z4;
    WTILE(subY,2) =   4.0f*z1 - 4.0f*z2 - z3 + z4;
    WTILE(subY,3) = - 2.0f*z1 - z2 + 2.0f*z3 + z4;
    WTILE(subY,4) =   2.0f*z1 - z2 - 2.0f*z3 + z4;
    WTILE(subY,5) = 4.0f*z1 - 5.0f*z3 + z5;
  }
#elif CONV_XSIZE == 5 && OUTTILE_XSIZE == 2
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    float z0 = WTILE(subY,0);
    float z1 = WTILE(subY,1);
    float z2 = WTILE(subY,2);
    float z3 = WTILE(subY,3);
    float z4 = WTILE(subY,4);
    float z5 = WTILE(subY,5);
    WTILE(subY,0) = 4.0f*z0 - 5.0f*z2 + z4;
    WTILE(subY,1) = - 4.0f*z1 - 4.0f*z2 + z3 + z4;
    WTILE(subY,2) =   4.0f*z1 - 4.0f*z2 - z3 + z4;
    WTILE(subY,3) = - 2.0f*z1 - z2 + 2.0f*z3 + z4;
    WTILE(subY,4) =   2.0f*z1 - z2 - 2.0f*z3 + z4;
    WTILE(subY,5) = 4.0f*z1 - 5.0f*z3 + z5;
  }
#else
  #error "No X winograd implemented for this conv and tile size"
#endif

#if CONV_YSIZE == 3 && OUTTILE_YSIZE == 2
  for(int subX = 0; subX < INTILE_XSIZE; subX++) {
    float z0 = WTILE(0,subX);
    float z1 = WTILE(1,subX);
    float z2 = WTILE(2,subX);
    float z3 = WTILE(3,subX);
    WTILE(0,subX) = z0 - z2;
    WTILE(1,subX) = z1 + z2;
    WTILE(2,subX) = z2 - z1;
    WTILE(3,subX) = z1 - z3;
  }
#elif CONV_YSIZE == 3 && OUTTILE_YSIZE == 4
  for(int subX = 0; subX < INTILE_XSIZE; subX++) {
    float z0 = WTILE(0,subX);
    float z1 = WTILE(1,subX);
    float z2 = WTILE(2,subX);
    float z3 = WTILE(3,subX);
    float z4 = WTILE(4,subX);
    float z5 = WTILE(5,subX);
    // Low error winograd
    // WTILE(0,subX) = z0 - 2.5f*z2 + z4;
    // WTILE(1,subX) = - SQRT2*z1 - 2.0f*z2 + SQRTHALF*z3 + z4;
    // WTILE(2,subX) =   SQRT2*z1 - 2.0f*z2 - SQRTHALF*z3 + z4;
    // WTILE(3,subX) = - SQRTHALF*z1 - 0.5f*z2 + SQRT2*z3 + z4;
    // WTILE(4,subX) =   SQRTHALF*z1 - 0.5f*z2 - SQRT2*z3 + z4;
    // WTILE(5,subX) = z1 - 2.5f*z3 + z5;
    WTILE(0,subX) = 4.0f*z0 - 5.0f*z2 + z4;
    WTILE(1,subX) = - 4.0f*z1 - 4.0f*z2 + z3 + z4;
    WTILE(2,subX) =   4.0f*z1 - 4.0f*z2 - z3 + z4;
    WTILE(3,subX) = - 2.0f*z1 - z2 + 2.0f*z3 + z4;
    WTILE(4,subX) =   2.0f*z1 - z2 - 2.0f*z3 + z4;
    WTILE(5,subX) = 4.0f*z1 - 5.0f*z3 + z5;
  }
#elif CONV_YSIZE == 5 && OUTTILE_YSIZE == 2
  for(int subX = 0; subX < INTILE_XSIZE; subX++) {
    float z0 = WTILE(0,subX);
    float z1 = WTILE(1,subX);
    float z2 = WTILE(2,subX);
    float z3 = WTILE(3,subX);
    float z4 = WTILE(4,subX);
    float z5 = WTILE(5,subX);
    WTILE(0,subX) = 4.0f*z0 - 5.0f*z2 + z4;
    WTILE(1,subX) = - 4.0f*z1 - 4.0f*z2 + z3 + z4;
    WTILE(2,subX) =   4.0f*z1 - 4.0f*z2 - z3 + z4;
    WTILE(3,subX) = - 2.0f*z1 - z2 + 2.0f*z3 + z4;
    WTILE(4,subX) =   2.0f*z1 - z2 - 2.0f*z3 + z4;
    WTILE(5,subX) = 4.0f*z1 - 5.0f*z3 + z5;
  }
#else
  #error "No Y winograd implemented for this conv and tile size"
#endif

#define TRANS(_suby,_subx,_ic,_ntile) transformed[(((_suby) * INTILE_XSIZE + (_subx))*icSize + (_ic)) * ntxtySize + (_ntile)]

  if(tileX < numTilesX && tileY < numTilesY && n < nSize && ic < icSize) {
    const int ntxtySize = nSize * numTilesX * numTilesY;
    const int ntile = (n * numTilesY + tileY) * numTilesX + tileX;

    //Copy private tile out to transformed output
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      for(int subX = 0; subX < INTILE_XSIZE; subX++) {
        TRANS(subY,subX,ic,ntile) = WTILE(subY,subX);
      }
    }
  }

}


__kernel void untransform(
  __global float* restrict transformed, //INTILE_YSIZE, INTILE_XSIZE, oc, batch, tileY, tileX
  __global float* restrict output,  //N, oc, H, W
  int nSize,
  int xSize,
  int ySize,
  int numTilesX,
  int numTilesY,
  int ocSize
) {
  const int tileX = get_global_id(0);
  const int tileY = get_global_id(1);
  const int noc = get_global_id(2);
  const int n = noc / ocSize;
  const int oc = noc % ocSize;

  const int ntxtySize = nSize * numTilesX * numTilesY;
  const int ntile = (n * numTilesY + tileY) * numTilesX + tileX;

#define WTILE(_y,_x) wTile[(_y)*INTILE_XSIZE + (_x)]
#define TRANS(_suby,_subx,_oc,_ntile) transformed[(((_suby) * INTILE_XSIZE + (_subx))*ocSize + (_oc)) * ntxtySize + (_ntile)]
#define OUTPUT(_noc,_y,_x) output[((_noc) * ySize + (_y)) * xSize + (_x)]

  __private float wTile[INTILE_XSIZE * INTILE_YSIZE];

  //Copy into private tile
  if(tileX < numTilesX && tileY < numTilesY && n < nSize) {
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      for(int subX = 0; subX < INTILE_XSIZE; subX++) {
        WTILE(subY,subX) = TRANS(subY,subX,oc,ntile);
      }
    }
  }

#if CONV_XSIZE == 3 && OUTTILE_XSIZE == 2
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    float z0 = WTILE(subY,0);
    float z1 = WTILE(subY,1);
    float z2 = WTILE(subY,2);
    float z3 = WTILE(subY,3);
    WTILE(subY,0) = z0 + z1 + z2;
    WTILE(subY,1) = z1 - z2 - z3;
  }
#elif CONV_XSIZE == 3 && OUTTILE_XSIZE == 4
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    float z0 = WTILE(subY,0);
    float z1 = WTILE(subY,1);
    float z2 = WTILE(subY,2);
    float z3 = WTILE(subY,3);
    float z4 = WTILE(subY,4);
    float z5 = WTILE(subY,5);
    WTILE(subY,0) = z0 + z1 + z2 + z3 + z4;
    // Low error winograd
    // WTILE(subY,1) = SQRTHALF*(z1 - z2) + SQRT2*(z3 - z4);
    // WTILE(subY,2) = 0.5f*(z1 + z2) + 2.0f*(z3 + z4);
    // WTILE(subY,3) = SQRTEIGHTH*(z1 - z2) + SQRT8*(z3 - z4) + z5;
    WTILE(subY,1) = (z1 - z2) + 2.0f*(z3 - z4);
    WTILE(subY,2) = (z1 + z2) + 4.0f*(z3 + z4);
    WTILE(subY,3) = (z1 - z2) + 8.0f*(z3 - z4) + z5;
  }
#elif CONV_XSIZE == 5 && OUTTILE_XSIZE == 2
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    float z0 = WTILE(subY,0);
    float z1 = WTILE(subY,1);
    float z2 = WTILE(subY,2);
    float z3 = WTILE(subY,3);
    float z4 = WTILE(subY,4);
    float z5 = WTILE(subY,5);
    WTILE(subY,0) = z0 + z1 + z2 + z3 + z4;
    WTILE(subY,1) = (z1 - z2) + 2.0f*(z3 - z4) + z5;
  }
#else
  #error "No X winograd implemented for this conv and tile size"
#endif

#if CONV_YSIZE == 3 && OUTTILE_YSIZE == 2
  for(int subX = 0; subX < OUTTILE_XSIZE; subX++) {
    float z0 = WTILE(0,subX);
    float z1 = WTILE(1,subX);
    float z2 = WTILE(2,subX);
    float z3 = WTILE(3,subX);
    WTILE(0,subX) = z0 + z1 + z2;
    WTILE(1,subX) = z1 - z2 - z3;
  }
#elif CONV_YSIZE == 3 && OUTTILE_YSIZE == 4
  for(int subX = 0; subX < OUTTILE_XSIZE; subX++) {
    float z0 = WTILE(0,subX);
    float z1 = WTILE(1,subX);
    float z2 = WTILE(2,subX);
    float z3 = WTILE(3,subX);
    float z4 = WTILE(4,subX);
    float z5 = WTILE(5,subX);
    WTILE(0,subX) = z0 + z1 + z2 + z3 + z4;
    // Low error winograd
    // WTILE(1,subX) = SQRTHALF*(z1 - z2) + SQRT2*(z3 - z4);
    // WTILE(2,subX) = 0.5f*(z1 + z2) + 2.0f*(z3 + z4);
    // WTILE(3,subX) = SQRTEIGHTH*(z1 - z2) + SQRT8*(z3 - z4) + z5;
    WTILE(1,subX) = (z1 - z2) + 2.0f*(z3 - z4);
    WTILE(2,subX) = (z1 + z2) + 4.0f*(z3 + z4);
    WTILE(3,subX) = (z1 - z2) + 8.0f*(z3 - z4) + z5;
  }
#elif CONV_YSIZE == 5 && OUTTILE_YSIZE == 2
  for(int subX = 0; subX < OUTTILE_XSIZE; subX++) {
    float z0 = WTILE(0,subX);
    float z1 = WTILE(1,subX);
    float z2 = WTILE(2,subX);
    float z3 = WTILE(3,subX);
    float z4 = WTILE(4,subX);
    float z5 = WTILE(5,subX);
    WTILE(0,subX) = z0 + z1 + z2 + z3 + z4;
    WTILE(1,subX) = (z1 - z2) + 2.0f*(z3 - z4) + z5;
  }
#else
  #error "No Y winograd implemented for this conv and tile size"
#endif

  //Copy into output
  for(int subY = 0; subY < OUTTILE_YSIZE; subY++) {
    int y = tileY * OUTTILE_YSIZE + subY;
    for(int subX = 0; subX < OUTTILE_XSIZE; subX++) {
      int x = tileX * OUTTILE_XSIZE + subX;
      if(y >= 0 && y < ySize && x >= 0 && x < xSize && tileX < numTilesX && tileY < numTilesY && n < nSize) {
        OUTPUT(noc,y,x) = WTILE(subY,subX);
      }
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

string OpenCLKernels::sumChannelsNCHW = R"%%(
//Defines:
//XYSTRIDE - power of two parallelism stride for reduction, should be get_local_size(0)
//CHANNELSTRIDE - stride for channels, should be get_local_size(1)
//LOCALSIZE_TOTAL - should be get_local_size(0) * get_local_size(1) * get_local_size(2)

//PRECONDIION: Kernel is being run where get_num_groups(0) == 1, so that global id and local id are identical for dim 0

__kernel void sumChannelsNCHW(
  __global float* input,  //N, c, HW
  __global float* output, //N, c
  int nSize,
  int cSize,
  int xySize
) {
  const int xyBase = get_local_id(0);
  const int c = get_global_id(1);
  const int n = get_global_id(2);
  const int localId1 = get_local_id(1);
  const int localId2 = get_local_id(2);

  __local float partialSums[LOCALSIZE_TOTAL];

  float sum = 0.0f;
  if(n < nSize && c < cSize) {
    //Sum up the elements that this group member is responsible for
    for(int xy = xyBase; xy < xySize; xy += XYSTRIDE) {
      int idx = (n * cSize + c) * xySize + xy;
      float v = input[idx];
      sum += v;
    }
  }

  //Write to local memory for performing the reduction
  int localIdx = (localId2 * CHANNELSTRIDE + localId1) * XYSTRIDE + xyBase;
  partialSums[localIdx] = sum;

  //Parallel folding downward
  for(int span = XYSTRIDE / 2; span > 0; span /= 2) {
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
//Defines:
//XYSTRIDE - power of two parallelism stride for reduction, should be get_local_size(0)
//CHANNELSTRIDE - stride for channels, should be get_local_size(1)
//LOCALSIZE_TOTAL - should be get_local_size(0) * get_local_size(1) * get_local_size(2)

//PRECONDIION: Kernel is being run where get_num_groups(0) == 1, so that global id and local id are identical for dim 0

__kernel void gPoolChannelsNCHW(
  __global float* input,  //N, c, HW
  __global float* output, //N, c
  __global float* maskSums, //N
  int nSize,
  int cSize,
  int xySize
) {
  const int xyBase = get_local_id(0);
  const int c = get_global_id(1);
  const int n = get_global_id(2);
  const int localId1 = get_local_id(1);
  const int localId2 = get_local_id(2);

  __local float partialSums[LOCALSIZE_TOTAL];
  __local float partialMaxes[LOCALSIZE_TOTAL];

  float sum = 0.0f;
  float max = 0.0f;
  if(n < nSize && c < cSize) {
    //Sum up the elements that this group member is responsible for
    for(int xy = xyBase; xy < xySize; xy += XYSTRIDE) {
      int idx = (n * cSize + c) * xySize + xy;
      float v = input[idx];
      sum += v;
      max = fmax(max,v);
    }
  }

  //Write to local memory for performing the reduction
  int localIdx = (localId2 * CHANNELSTRIDE + localId1) * XYSTRIDE + xyBase;
  partialSums[localIdx] = sum;
  partialMaxes[localIdx] = max;

  //Parallel folding downward
  for(int span = XYSTRIDE / 2; span > 0; span /= 2) {
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
//Defines:
//XYSTRIDE - power of two parallelism stride for reduction, should be get_local_size(0)
//CHANNELSTRIDE - stride for channels, should be get_local_size(1)
//LOCALSIZE_TOTAL - should be get_local_size(0) * get_local_size(1) * get_local_size(2)

//PRECONDIION: Kernel is being run where get_num_groups(0) == 1, so that global id and local id are identical for dim 0

__kernel void valueHeadPoolChannelsNCHW(
  __global float* input,  //N, c, HW
  __global float* output, //N, c
  __global float* maskSums, //N
  int nSize,
  int cSize,
  int xySize
) {
  const int xyBase = get_local_id(0);
  const int c = get_global_id(1);
  const int n = get_global_id(2);
  const int localId1 = get_local_id(1);
  const int localId2 = get_local_id(2);

  __local float partialSums[LOCALSIZE_TOTAL];

  float sum = 0.0f;
  if(n < nSize && c < cSize) {
    //Sum up the elements that this group member is responsible for
    for(int xy = xyBase; xy < xySize; xy += XYSTRIDE) {
      int idx = (n * cSize + c) * xySize + xy;
      float v = input[idx];
      sum += v;
    }
  }

  //Write to local memory for performing the reduction
  int localIdx = (localId2 * CHANNELSTRIDE + localId1) * XYSTRIDE + xyBase;
  partialSums[localIdx] = sum;

  //Parallel folding downward
  for(int span = XYSTRIDE / 2; span > 0; span /= 2) {
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
//Defines
//TILEDIM - width and height of tile to use for transposing
//TILESTRIDE - y step size across tile
//LOCALSIZE - TILEDIM * (TILEDIM+1)
//+1 avoids bank conflicts
//get_local_size(2) is assumed to be 1

__kernel void transposeNCHW(
  __global float* in,
  __global float* out,
  int xSize,
  int ySize,
  int ncSize
) {
  const int tileDimP1 = TILEDIM+1;
  const int xLocal = get_local_id(0);
  const int yLocal = get_local_id(1);
  const int xIdx = get_global_id(0);
  const int yBase = get_group_id(1) * TILEDIM;
  const int nc = get_global_id(2);
  const int xySize = xSize * ySize;

  __local float tileNCHW[LOCALSIZE];

  if(xIdx < xSize && nc < ncSize) {
    for(int j = yLocal; j < TILEDIM && yBase+j < ySize; j += TILESTRIDE) {
      int inIdx = xIdx + xSize * (yBase+j) + xySize * nc;
      tileNCHW[j*tileDimP1 + xLocal] = in[inIdx];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  //Transpose idx
  int outXIdx = get_group_id(1) * TILEDIM + xLocal;
  int outYBase = get_group_id(0) * TILEDIM;

  if(outXIdx < ySize && nc < ncSize) {
    for(int j = yLocal; j < TILEDIM && outYBase+j < xSize; j += TILESTRIDE) {
      int outIdx = outXIdx + ySize * (outYBase+j) + xySize * nc;
      out[outIdx] = tileNCHW[xLocal*tileDimP1 + j];
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


string OpenCLKernels::xgemmDirect =
"#define ROUTINE_GEMMBATCHED\n"
"#define ROUTINE_GEMMSTRIDEDBATCHED\n"
#include "../external/clblast/common.opencl"
#include "../external/clblast/xgemm_direct_part1.opencl"
#include "../external/clblast/xgemm_direct_part2.opencl"
#include "../external/clblast/xgemm_direct_part3.opencl"
#include "../external/clblast/xgemm_direct_batched.opencl"
;

string OpenCLKernels::xgemm =
"#define ROUTINE_GEMMBATCHED\n"
"#define ROUTINE_GEMMSTRIDEDBATCHED\n"
#include "../external/clblast/common.opencl"
#include "../external/clblast/xgemm_part1.opencl"
#include "../external/clblast/xgemm_part2.opencl"
#include "../external/clblast/xgemm_part3.opencl"
#include "../external/clblast/xgemm_batched.opencl"
;

#endif
