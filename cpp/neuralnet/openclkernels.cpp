#ifdef USE_OPENCL_BACKEND

#include "../neuralnet/openclkernels.h"

using namespace std;

string OpenCLKernels::common = R"%%(
#ifndef PRECISION
  #define PRECISION 32
#endif
#ifndef PRECISION_STORAGE
  #define PRECISION_STORAGE 32
#endif

#if PRECISION == 16
  #pragma OPENCL EXTENSION cl_khr_fp16: enable
  typedef half real;
  #define ZERO 0.0h
  #define ONE 1.0h
  #define HUNDRED 100.0h
  #define FOURTEEN 14.0h
  #define TEN 10.0h
  #define EIGHT 8.0h
  #define FIVE 5.0h
  #define FOUR 4.0h
  #define TWO 2.0h
  #define HALF 0.5h
  #define TWOP5 2.5h
  #define SQRT8 2.82842712475h
  #define SQRT2 1.41421356237h
  #define SQRTHALF 0.70710678118h
  #define SQRTEIGHTH 0.35355339059h
  #define floatToReal(_r) (convert_half(_r))

#elif PRECISION == 32
  typedef float real;
  #define ZERO 0.0f
  #define ONE 1.0f
  #define HUNDRED 100.0f
  #define FOURTEEN 14.0f
  #define TEN 10.0f
  #define EIGHT 8.0f
  #define FIVE 5.0f
  #define FOUR 4.0f
  #define TWO 2.0f
  #define HALF 0.5f
  #define TWOP5 2.5f
  #define SQRT8 2.82842712475f
  #define SQRT2 1.41421356237f
  #define SQRTHALF 0.70710678118f
  #define SQRTEIGHTH 0.35355339059f
  #define floatToReal(_r) (_r)
#endif

#if PRECISION_STORAGE == 16
  typedef half realstore;
  #if PRECISION == 16
    #define LOAD(__buf,__x) ((__buf)[(__x)])
    #define STORE(__buf,__x,__y) ((__buf)[(__x)] = (__y))
  #elif PRECISION == 32
    #define LOAD(__buf,__x) vload_half((__x),(__buf))
    #define STORE(__buf,__x,__y) vstore_half((__y),(__x),(__buf))
  #endif
#elif PRECISION_STORAGE == 32
  typedef float realstore;
  #define LOAD(__buf,__x) ((__buf)[(__x)])
  #define STORE(__buf,__x,__y) ((__buf)[(__x)] = (__y))
#endif

)%%";

string OpenCLKernels::conv2dNCHW = OpenCLKernels::common + R"%%(

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
  __global realstore* restrict input,  //N, ic, H, W
  __global realstore* restrict filter, //oc, ic, fy, fx
  __global realstore* restrict output, //N, oc, H, W

  __local real* restrict inputTile, //ic, H, W      size = TILE_CHANNELS * inputTileXSize * inputTileYSize
  __local real* restrict outputTile, //H, W         size = TILE_XSIZE * TILE_YSIZE

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

#define INPUT(_n,_ic,_y,_x) LOAD(input,((_n) * icSize + (_ic)) * xySize + (_y) * xSize + (_x))
#define INPUTTILE(_ic,_ity,_itx) inputTile[((_ic) * inputTileYSize + (_ity)) * inputTileXSize + (_itx)]

#define FILTER(_oc,_ic,_y,_x) LOAD(filter,(((_oc) * icSize + (_ic)) * fySize + (_y)) * fxSize + (_x))

#define WRITEOUTPUT(_n,_oc,_y,_x,_value) STORE(output,((_n) * ocSize + (_oc)) * xySize + (_y) * xSize + (_x),_value)
#define OUTPUTTILE(_oty,_otx) outputTile[(_oty) * TILE_XSIZE + (_otx)]

  for(int n = 0; n < nSize; n++) {
    real acc = ZERO;

    //Initialize outputTile. No need to sync for this tile since each thread only ever reads its own spots
    for(int oty = ly; oty<TILE_YSIZE; oty += lySize) {
      for(int otx = lx; otx<TILE_XSIZE; otx += lxSize) {
        OUTPUTTILE(oty,otx) = ZERO;
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
            real inputValue = ZERO;
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
          real acc = ZERO;
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
          real result = OUTPUTTILE(oty,otx);
          WRITEOUTPUT(n, oc, yBase+oty, xBase+otx, result);
        }
      }
    }
  } //Close loop over batch
}

)%%";


string OpenCLKernels::winogradTransformNCHW = OpenCLKernels::common + R"%%(

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

__kernel void transform(
  __global realstore* restrict input,  //N, ic, H, W
  __global realstore* restrict transformed, //(INTILE_YSIZE, INTILE_XSIZE), (ic), (batch, tileY, tileX) where the last two dimenions are padded
  int nSize,
  int xSize,
  int ySize,
  int numTilesX,
  int numTilesY,
  int icSize,
  int icSizePadded,
  int ntxtySizePadded
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

#define INPUT(_nic,_xy) LOAD(input,((_nic) * xySize) + (_xy))
#define WTILE(_y,_x) wTile[(_y)*INTILE_XSIZE + (_x)]

  __private real wTile[INTILE_XSIZE * INTILE_YSIZE];

  //Copy input into private tile
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    int y = tileY * OUTTILE_YSIZE + subY + INTILE_YOFFSET;
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      int x = tileX * OUTTILE_XSIZE + subX + INTILE_XOFFSET;
      real value = ZERO;
      if(y >= 0 && y < ySize && x >= 0 && x < xSize && n < nSize && ic < icSize) {
        int xy = y * xSize + x;
        value = INPUT(nic,xy);
      }
      WTILE(subY,subX) = value;
    }
  }

#if CONV_XSIZE == 3 && OUTTILE_XSIZE == 2
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    real z0 = WTILE(subY,0);
    real z1 = WTILE(subY,1);
    real z2 = WTILE(subY,2);
    real z3 = WTILE(subY,3);
    WTILE(subY,0) = z0 - z2;
    WTILE(subY,1) = z1 + z2;
    WTILE(subY,2) = z2 - z1;
    WTILE(subY,3) = z1 - z3;
  }
#elif CONV_XSIZE == 3 && OUTTILE_XSIZE == 4
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    real z0 = WTILE(subY,0);
    real z1 = WTILE(subY,1);
    real z2 = WTILE(subY,2);
    real z3 = WTILE(subY,3);
    real z4 = WTILE(subY,4);
    real z5 = WTILE(subY,5);
    // Low error winograd
    // WTILE(subY,0) = z0 - TWOP5*z2 + z4;
    // WTILE(subY,1) = - SQRT2*z1 - TWO*z2 + SQRTHALF*z3 + z4;
    // WTILE(subY,2) =   SQRT2*z1 - TWO*z2 - SQRTHALF*z3 + z4;
    // WTILE(subY,3) = - SQRTHALF*z1 - HALF*z2 + SQRT2*z3 + z4;
    // WTILE(subY,4) =   SQRTHALF*z1 - HALF*z2 - SQRT2*z3 + z4;
    // WTILE(subY,5) = z1 - TWOP5*z3 + z5;
    WTILE(subY,0) = FOUR*z0 - FIVE*z2 + z4;
    WTILE(subY,1) = - FOUR*z1 - FOUR*z2 + z3 + z4;
    WTILE(subY,2) =   FOUR*z1 - FOUR*z2 - z3 + z4;
    WTILE(subY,3) = - TWO*z1 - z2 + TWO*z3 + z4;
    WTILE(subY,4) =   TWO*z1 - z2 - TWO*z3 + z4;
    WTILE(subY,5) = FOUR*z1 - FIVE*z3 + z5;
  }
#elif CONV_XSIZE == 5 && OUTTILE_XSIZE == 2
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    real z0 = WTILE(subY,0);
    real z1 = WTILE(subY,1);
    real z2 = WTILE(subY,2);
    real z3 = WTILE(subY,3);
    real z4 = WTILE(subY,4);
    real z5 = WTILE(subY,5);
    WTILE(subY,0) = FOUR*z0 - FIVE*z2 + z4;
    WTILE(subY,1) = - FOUR*z1 - FOUR*z2 + z3 + z4;
    WTILE(subY,2) =   FOUR*z1 - FOUR*z2 - z3 + z4;
    WTILE(subY,3) = - TWO*z1 - z2 + TWO*z3 + z4;
    WTILE(subY,4) =   TWO*z1 - z2 - TWO*z3 + z4;
    WTILE(subY,5) = FOUR*z1 - FIVE*z3 + z5;
  }
#else
  #error "No X winograd implemented for this conv and tile size"
#endif

#if CONV_YSIZE == 3 && OUTTILE_YSIZE == 2
  for(int subX = 0; subX < INTILE_XSIZE; subX++) {
    real z0 = WTILE(0,subX);
    real z1 = WTILE(1,subX);
    real z2 = WTILE(2,subX);
    real z3 = WTILE(3,subX);
    WTILE(0,subX) = z0 - z2;
    WTILE(1,subX) = z1 + z2;
    WTILE(2,subX) = z2 - z1;
    WTILE(3,subX) = z1 - z3;
  }
#elif CONV_YSIZE == 3 && OUTTILE_YSIZE == 4
  for(int subX = 0; subX < INTILE_XSIZE; subX++) {
    real z0 = WTILE(0,subX);
    real z1 = WTILE(1,subX);
    real z2 = WTILE(2,subX);
    real z3 = WTILE(3,subX);
    real z4 = WTILE(4,subX);
    real z5 = WTILE(5,subX);
    // Low error winograd
    // WTILE(0,subX) = z0 - TWOP5*z2 + z4;
    // WTILE(1,subX) = - SQRT2*z1 - TWO*z2 + SQRTHALF*z3 + z4;
    // WTILE(2,subX) =   SQRT2*z1 - TWO*z2 - SQRTHALF*z3 + z4;
    // WTILE(3,subX) = - SQRTHALF*z1 - HALF*z2 + SQRT2*z3 + z4;
    // WTILE(4,subX) =   SQRTHALF*z1 - HALF*z2 - SQRT2*z3 + z4;
    // WTILE(5,subX) = z1 - TWOP5*z3 + z5;
    WTILE(0,subX) = FOUR*z0 - FIVE*z2 + z4;
    WTILE(1,subX) = - FOUR*z1 - FOUR*z2 + z3 + z4;
    WTILE(2,subX) =   FOUR*z1 - FOUR*z2 - z3 + z4;
    WTILE(3,subX) = - TWO*z1 - z2 + TWO*z3 + z4;
    WTILE(4,subX) =   TWO*z1 - z2 - TWO*z3 + z4;
    WTILE(5,subX) = FOUR*z1 - FIVE*z3 + z5;
  }
#elif CONV_YSIZE == 5 && OUTTILE_YSIZE == 2
  for(int subX = 0; subX < INTILE_XSIZE; subX++) {
    real z0 = WTILE(0,subX);
    real z1 = WTILE(1,subX);
    real z2 = WTILE(2,subX);
    real z3 = WTILE(3,subX);
    real z4 = WTILE(4,subX);
    real z5 = WTILE(5,subX);
    WTILE(0,subX) = FOUR*z0 - FIVE*z2 + z4;
    WTILE(1,subX) = - FOUR*z1 - FOUR*z2 + z3 + z4;
    WTILE(2,subX) =   FOUR*z1 - FOUR*z2 - z3 + z4;
    WTILE(3,subX) = - TWO*z1 - z2 + TWO*z3 + z4;
    WTILE(4,subX) =   TWO*z1 - z2 - TWO*z3 + z4;
    WTILE(5,subX) = FOUR*z1 - FIVE*z3 + z5;
  }
#else
  #error "No Y winograd implemented for this conv and tile size"
#endif

#define WRITETRANS(_suby,_subx,_ic,_ntile,_value) STORE(transformed,(((_suby) * INTILE_XSIZE + (_subx))*icSizePadded + (_ic))*ntxtySizePadded + (_ntile),_value)

  if(ntxty < ntxtySizePadded && ic < icSizePadded) {
    //Copy private tile out to transformed output
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      for(int subX = 0; subX < INTILE_XSIZE; subX++) {
        real result = WTILE(subY,subX);
        WRITETRANS(subY,subX,ic,ntxty,result);
      }
    }
  }

}

)%%";

string OpenCLKernels::winogradBNReluTransformNCHW = OpenCLKernels::common + R"%%(

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

__kernel void bnReluTransform(
  __global realstore* restrict input,  //N, ic, H, W
  __global realstore* restrict transformed, //(INTILE_YSIZE, INTILE_XSIZE), (ic), (batch, tileY, tileX) where the last two dimenions are padded
  __global realstore* restrict scale, //ic
  __global realstore* restrict bias, //ic
  __global realstore* restrict mask, //N, H, W
  int nSize,
  int xSize,
  int ySize,
  int numTilesX,
  int numTilesY,
  int icSize,
  int icSizePadded,
  int ntxtySizePadded
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

#define INPUT(_nic,_xy) LOAD(input,((_nic) * xySize) + (_xy))
#define WTILE(_y,_x) wTile[(_y)*INTILE_XSIZE + (_x)]

  __private real wTile[INTILE_XSIZE * INTILE_YSIZE];

  //Copy input into private tile
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    int y = tileY * OUTTILE_YSIZE + subY + INTILE_YOFFSET;
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      int x = tileX * OUTTILE_XSIZE + subX + INTILE_XOFFSET;
      real value = ZERO;
      if(y >= 0 && y < ySize && x >= 0 && x < xSize && n < nSize && ic < icSize) {
        int xy = y * xSize + x;
        value = fmax(INPUT(nic,xy) * LOAD(scale,ic) + LOAD(bias,ic), ZERO) * LOAD(mask, n * xySize + xy);
      }
      WTILE(subY,subX) = value;
    }
  }

#if CONV_XSIZE == 3 && OUTTILE_XSIZE == 2
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    real z0 = WTILE(subY,0);
    real z1 = WTILE(subY,1);
    real z2 = WTILE(subY,2);
    real z3 = WTILE(subY,3);
    WTILE(subY,0) = z0 - z2;
    WTILE(subY,1) = z1 + z2;
    WTILE(subY,2) = z2 - z1;
    WTILE(subY,3) = z1 - z3;
  }
#elif CONV_XSIZE == 3 && OUTTILE_XSIZE == 4
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    real z0 = WTILE(subY,0);
    real z1 = WTILE(subY,1);
    real z2 = WTILE(subY,2);
    real z3 = WTILE(subY,3);
    real z4 = WTILE(subY,4);
    real z5 = WTILE(subY,5);
    // Low error winograd
    // WTILE(subY,0) = z0 - TWOP5*z2 + z4;
    // WTILE(subY,1) = - SQRT2*z1 - TWO*z2 + SQRTHALF*z3 + z4;
    // WTILE(subY,2) =   SQRT2*z1 - TWO*z2 - SQRTHALF*z3 + z4;
    // WTILE(subY,3) = - SQRTHALF*z1 - HALF*z2 + SQRT2*z3 + z4;
    // WTILE(subY,4) =   SQRTHALF*z1 - HALF*z2 - SQRT2*z3 + z4;
    // WTILE(subY,5) = z1 - TWOP5*z3 + z5;
    WTILE(subY,0) = FOUR*z0 - FIVE*z2 + z4;
    WTILE(subY,1) = - FOUR*z1 - FOUR*z2 + z3 + z4;
    WTILE(subY,2) =   FOUR*z1 - FOUR*z2 - z3 + z4;
    WTILE(subY,3) = - TWO*z1 - z2 + TWO*z3 + z4;
    WTILE(subY,4) =   TWO*z1 - z2 - TWO*z3 + z4;
    WTILE(subY,5) = FOUR*z1 - FIVE*z3 + z5;
  }
#elif CONV_XSIZE == 5 && OUTTILE_XSIZE == 2
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    real z0 = WTILE(subY,0);
    real z1 = WTILE(subY,1);
    real z2 = WTILE(subY,2);
    real z3 = WTILE(subY,3);
    real z4 = WTILE(subY,4);
    real z5 = WTILE(subY,5);
    WTILE(subY,0) = FOUR*z0 - FIVE*z2 + z4;
    WTILE(subY,1) = - FOUR*z1 - FOUR*z2 + z3 + z4;
    WTILE(subY,2) =   FOUR*z1 - FOUR*z2 - z3 + z4;
    WTILE(subY,3) = - TWO*z1 - z2 + TWO*z3 + z4;
    WTILE(subY,4) =   TWO*z1 - z2 - TWO*z3 + z4;
    WTILE(subY,5) = FOUR*z1 - FIVE*z3 + z5;
  }
#else
  #error "No X winograd implemented for this conv and tile size"
#endif

#if CONV_YSIZE == 3 && OUTTILE_YSIZE == 2
  for(int subX = 0; subX < INTILE_XSIZE; subX++) {
    real z0 = WTILE(0,subX);
    real z1 = WTILE(1,subX);
    real z2 = WTILE(2,subX);
    real z3 = WTILE(3,subX);
    WTILE(0,subX) = z0 - z2;
    WTILE(1,subX) = z1 + z2;
    WTILE(2,subX) = z2 - z1;
    WTILE(3,subX) = z1 - z3;
  }
#elif CONV_YSIZE == 3 && OUTTILE_YSIZE == 4
  for(int subX = 0; subX < INTILE_XSIZE; subX++) {
    real z0 = WTILE(0,subX);
    real z1 = WTILE(1,subX);
    real z2 = WTILE(2,subX);
    real z3 = WTILE(3,subX);
    real z4 = WTILE(4,subX);
    real z5 = WTILE(5,subX);
    // Low error winograd
    // WTILE(0,subX) = z0 - TWOP5*z2 + z4;
    // WTILE(1,subX) = - SQRT2*z1 - TWO*z2 + SQRTHALF*z3 + z4;
    // WTILE(2,subX) =   SQRT2*z1 - TWO*z2 - SQRTHALF*z3 + z4;
    // WTILE(3,subX) = - SQRTHALF*z1 - HALF*z2 + SQRT2*z3 + z4;
    // WTILE(4,subX) =   SQRTHALF*z1 - HALF*z2 - SQRT2*z3 + z4;
    // WTILE(5,subX) = z1 - TWOP5*z3 + z5;
    WTILE(0,subX) = FOUR*z0 - FIVE*z2 + z4;
    WTILE(1,subX) = - FOUR*z1 - FOUR*z2 + z3 + z4;
    WTILE(2,subX) =   FOUR*z1 - FOUR*z2 - z3 + z4;
    WTILE(3,subX) = - TWO*z1 - z2 + TWO*z3 + z4;
    WTILE(4,subX) =   TWO*z1 - z2 - TWO*z3 + z4;
    WTILE(5,subX) = FOUR*z1 - FIVE*z3 + z5;
  }
#elif CONV_YSIZE == 5 && OUTTILE_YSIZE == 2
  for(int subX = 0; subX < INTILE_XSIZE; subX++) {
    real z0 = WTILE(0,subX);
    real z1 = WTILE(1,subX);
    real z2 = WTILE(2,subX);
    real z3 = WTILE(3,subX);
    real z4 = WTILE(4,subX);
    real z5 = WTILE(5,subX);
    WTILE(0,subX) = FOUR*z0 - FIVE*z2 + z4;
    WTILE(1,subX) = - FOUR*z1 - FOUR*z2 + z3 + z4;
    WTILE(2,subX) =   FOUR*z1 - FOUR*z2 - z3 + z4;
    WTILE(3,subX) = - TWO*z1 - z2 + TWO*z3 + z4;
    WTILE(4,subX) =   TWO*z1 - z2 - TWO*z3 + z4;
    WTILE(5,subX) = FOUR*z1 - FIVE*z3 + z5;
  }
#else
  #error "No Y winograd implemented for this conv and tile size"
#endif

#define WRITETRANS(_suby,_subx,_ic,_ntile,_value) STORE(transformed,(((_suby) * INTILE_XSIZE + (_subx))*icSizePadded + (_ic))*ntxtySizePadded + (_ntile),_value)

  if(ntxty < ntxtySizePadded && ic < icSizePadded) {
    //Copy private tile out to transformed output
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      for(int subX = 0; subX < INTILE_XSIZE; subX++) {
        real result = WTILE(subY,subX);
        WRITETRANS(subY,subX,ic,ntxty,result);
      }
    }
  }

}

)%%";

string OpenCLKernels::winogradUntransformNCHW = OpenCLKernels::common + R"%%(

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

__kernel void untransform(
  __global realstore* restrict transformed, //(INTILE_YSIZE, INTILE_XSIZE), (oc), (batch, tileY, tileX) //where the last two dims are padded
  __global realstore* restrict output,  //N, oc, H, W
  int nSize,
  int xSize,
  int ySize,
  int numTilesX,
  int numTilesY,
  int ocSize,
  int ocSizePadded,
  int ntxtySizePadded
) {
  const int tileX = get_global_id(0);
  const int tileY = get_global_id(1);
  const int noc = get_global_id(2);
  const int n = noc / ocSize;
  const int oc = noc % ocSize;

  const int ntile = (n * numTilesY + tileY) * numTilesX + tileX;

#define WTILE(_y,_x) wTile[(_y)*INTILE_XSIZE + (_x)]
#define TRANS(_suby,_subx,_oc,_ntile) LOAD(transformed,(((_suby) * INTILE_XSIZE + (_subx))*ocSizePadded + (_oc)) * ntxtySizePadded + (_ntile))
#define WRITEOUTPUT(_noc,_y,_x,_value) STORE(output,((_noc) * ySize + (_y)) * xSize + (_x),_value)

  __private real wTile[INTILE_XSIZE * INTILE_YSIZE];

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
    real z0 = WTILE(subY,0);
    real z1 = WTILE(subY,1);
    real z2 = WTILE(subY,2);
    real z3 = WTILE(subY,3);
    WTILE(subY,0) = z0 + z1 + z2;
    WTILE(subY,1) = z1 - z2 - z3;
  }
#elif CONV_XSIZE == 3 && OUTTILE_XSIZE == 4
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    real z0 = WTILE(subY,0);
    real z1 = WTILE(subY,1);
    real z2 = WTILE(subY,2);
    real z3 = WTILE(subY,3);
    real z4 = WTILE(subY,4);
    real z5 = WTILE(subY,5);
    WTILE(subY,0) = z0 + z1 + z2 + z3 + z4;
    // Low error winograd
    // WTILE(subY,1) = SQRTHALF*(z1 - z2) + SQRT2*(z3 - z4);
    // WTILE(subY,2) = HALF*(z1 + z2) + TWO*(z3 + z4);
    // WTILE(subY,3) = SQRTEIGHTH*(z1 - z2) + SQRT8*(z3 - z4) + z5;
    WTILE(subY,1) = (z1 - z2) + TWO*(z3 - z4);
    WTILE(subY,2) = (z1 + z2) + FOUR*(z3 + z4);
    WTILE(subY,3) = (z1 - z2) + EIGHT*(z3 - z4) + z5;
  }
#elif CONV_XSIZE == 5 && OUTTILE_XSIZE == 2
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    real z0 = WTILE(subY,0);
    real z1 = WTILE(subY,1);
    real z2 = WTILE(subY,2);
    real z3 = WTILE(subY,3);
    real z4 = WTILE(subY,4);
    real z5 = WTILE(subY,5);
    WTILE(subY,0) = z0 + z1 + z2 + z3 + z4;
    WTILE(subY,1) = (z1 - z2) + TWO*(z3 - z4) + z5;
  }
#else
  #error "No X winograd implemented for this conv and tile size"
#endif

#if CONV_YSIZE == 3 && OUTTILE_YSIZE == 2
  for(int subX = 0; subX < OUTTILE_XSIZE; subX++) {
    real z0 = WTILE(0,subX);
    real z1 = WTILE(1,subX);
    real z2 = WTILE(2,subX);
    real z3 = WTILE(3,subX);
    WTILE(0,subX) = z0 + z1 + z2;
    WTILE(1,subX) = z1 - z2 - z3;
  }
#elif CONV_YSIZE == 3 && OUTTILE_YSIZE == 4
  for(int subX = 0; subX < OUTTILE_XSIZE; subX++) {
    real z0 = WTILE(0,subX);
    real z1 = WTILE(1,subX);
    real z2 = WTILE(2,subX);
    real z3 = WTILE(3,subX);
    real z4 = WTILE(4,subX);
    real z5 = WTILE(5,subX);
    WTILE(0,subX) = z0 + z1 + z2 + z3 + z4;
    // Low error winograd
    // WTILE(1,subX) = SQRTHALF*(z1 - z2) + SQRT2*(z3 - z4);
    // WTILE(2,subX) = HALF*(z1 + z2) + TWO*(z3 + z4);
    // WTILE(3,subX) = SQRTEIGHTH*(z1 - z2) + SQRT8*(z3 - z4) + z5;
    WTILE(1,subX) = (z1 - z2) + TWO*(z3 - z4);
    WTILE(2,subX) = (z1 + z2) + FOUR*(z3 + z4);
    WTILE(3,subX) = (z1 - z2) + EIGHT*(z3 - z4) + z5;
  }
#elif CONV_YSIZE == 5 && OUTTILE_YSIZE == 2
  for(int subX = 0; subX < OUTTILE_XSIZE; subX++) {
    real z0 = WTILE(0,subX);
    real z1 = WTILE(1,subX);
    real z2 = WTILE(2,subX);
    real z3 = WTILE(3,subX);
    real z4 = WTILE(4,subX);
    real z5 = WTILE(5,subX);
    WTILE(0,subX) = z0 + z1 + z2 + z3 + z4;
    WTILE(1,subX) = (z1 - z2) + TWO*(z3 - z4) + z5;
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
        real result = WTILE(subY,subX);
        WRITEOUTPUT(noc,y,x,result);
      }
    }
  }

}

)%%";


string OpenCLKernels::scaleBiasMaskNCHW = OpenCLKernels::common + R"%%(
__kernel void scaleBiasMaskNCHW(
  __global realstore* input,  //N, c, H, W
  __global realstore* output, //N, c, H, W
  __global realstore* scale,  //c
  __global realstore* bias,   //c
  __global realstore* mask,   //N, H, W
  int nSize,
  int cSize,
  int xySize
) {
  const int xy = get_global_id(0);
  const int c = get_global_id(1);

  if(c < cSize && xy < xySize) {
    for(int n = 0; n < nSize; n++) {
      int idx = (n * cSize + c) * xySize + xy;
      real result = (LOAD(input,idx) * LOAD(scale,c) + LOAD(bias,c)) * LOAD(mask,n * xySize + xy);
      STORE(output,idx,result);
    }
  }
}
)%%";

string OpenCLKernels::scaleBiasMaskReluNCHW = OpenCLKernels::common + R"%%(
__kernel void scaleBiasMaskReluNCHW(
  __global realstore* input,  //N, c, H, W
  __global realstore* output, //N, c, H, W
  __global realstore* scale,  //c
  __global realstore* bias,   //c
  __global realstore* mask,   //N, H, W
  int nSize,
  int cSize,
  int xySize
) {
  const int xy = get_global_id(0);
  const int c = get_global_id(1);

  if(c < cSize && xy < xySize) {
    for(int n = 0; n < nSize; n++) {
      int idx = (n * cSize + c) * xySize + xy;
      real result = fmax(LOAD(input,idx) * LOAD(scale,c) + LOAD(bias,c), ZERO) * LOAD(mask,n * xySize + xy);
      STORE(output,idx,result);
    }
  }
}
)%%";

string OpenCLKernels::addPointWise = OpenCLKernels::common + R"%%(
__kernel void addPointWise(
  __global realstore* accum,
  __global realstore* value,
  int size
) {
  const int s = get_global_id(0);

  if(s < size)
    accum[s] += value[s];
}
)%%";

string OpenCLKernels::sumChannelsNCHW = OpenCLKernels::common + R"%%(
//Defines:
//XYSTRIDE - power of two parallelism stride for reduction, should be get_local_size(0)
//CHANNELSTRIDE - stride for channels, should be get_local_size(1)
//LOCALSIZE_TOTAL - should be get_local_size(0) * get_local_size(1) * get_local_size(2)

//PRECONDIION: Kernel is being run where get_num_groups(0) == 1, so that global id and local id are identical for dim 0

__kernel void sumChannelsNCHW(
  __global realstore* input,  //N, c, HW
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
      float v = LOAD(input,idx);
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


string OpenCLKernels::gPoolChannelsNCHW = OpenCLKernels::common + R"%%(
//Defines:
//XYSTRIDE - power of two parallelism stride for reduction, should be get_local_size(0)
//CHANNELSTRIDE - stride for channels, should be get_local_size(1)
//LOCALSIZE_TOTAL - should be get_local_size(0) * get_local_size(1) * get_local_size(2)

//PRECONDIION: Kernel is being run where get_num_groups(0) == 1, so that global id and local id are identical for dim 0

__kernel void gPoolChannelsNCHW(
  __global realstore* input,  //N, c, HW
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
      float v = LOAD(input,idx);
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

string OpenCLKernels::valueHeadPoolChannelsNCHW = OpenCLKernels::common + R"%%(
//Defines:
//XYSTRIDE - power of two parallelism stride for reduction, should be get_local_size(0)
//CHANNELSTRIDE - stride for channels, should be get_local_size(1)
//LOCALSIZE_TOTAL - should be get_local_size(0) * get_local_size(1) * get_local_size(2)

//PRECONDIION: Kernel is being run where get_num_groups(0) == 1, so that global id and local id are identical for dim 0

__kernel void valueHeadPoolChannelsNCHW(
  __global realstore* input,  //N, c, HW
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
      float v = LOAD(input,idx);
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


string OpenCLKernels::addChannelBiasesNCHW = OpenCLKernels::common + R"%%(
__kernel void addChannelBiasesNCHW(
  __global realstore* accum,  //NC, HW
  __global float* biases, //NC
  int ncSize,
  int xySize
) {
  const int xy = get_global_id(0);
  const int nc = get_global_id(1);

  if(nc < ncSize && xy < xySize) {
    real result = LOAD(accum,nc * xySize + xy) + floatToReal(biases[nc]);
    STORE(accum, nc * xySize + xy, result);
  }
}
)%%";


string OpenCLKernels::addCBiasesNC = OpenCLKernels::common + R"%%(
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


string OpenCLKernels::addCBiasesNCRelu = OpenCLKernels::common + R"%%(
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


string OpenCLKernels::extractChannel0NCHW = OpenCLKernels::common + R"%%(
__kernel void extractChannel0NCHW(__global realstore* in, __global realstore* out, int nSize, int cSize, int xySize)
{
  const int xyIdx = get_global_id(0);
  const int nIdx = get_global_id(1);
  if(xyIdx < xySize && nIdx < nSize) {
    out[nIdx * xySize + xyIdx] = in[nIdx * cSize * xySize + xyIdx];
  }
}
)%%";

//!
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
