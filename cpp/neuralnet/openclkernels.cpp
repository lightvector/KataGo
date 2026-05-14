#ifdef USE_OPENCL_BACKEND

#include "../neuralnet/openclkernels.h"

using namespace std;

string OpenCLKernels::fp16StorageDefine = " -DPRECISION_STORAGE=16";
string OpenCLKernels::fp16ComputeDefine = " -DPRECISION=16";

string OpenCLKernels::actIdenDefine = " -DACTIVATION=0";
string OpenCLKernels::actReluDefine = " -DACTIVATION=1";
string OpenCLKernels::actMishDefine = " -DACTIVATION=2";
string OpenCLKernels::actMishScale8Define = " -DACTIVATION=12";
string OpenCLKernels::actSiluDefine = " -DACTIVATION=3";

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
  #define LOG1PEXPTHRESHOLD 20.0f
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
  #define LOG1PEXPTHRESHOLD 20.0f
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
  int filterYRadius,
  int xyStride // Stride between spatial planes (may be > xSize*ySize due to padding)
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

  const int fxSize = (2 * filterXRadius + 1);
  const int fySize = (2 * filterYRadius + 1);

#define INPUT(_n,_ic,_y,_x) LOAD(input,((_n) * icSize + (_ic)) * xyStride + (_y) * xSize + (_x))
#define INPUTTILE(_ic,_ity,_itx) inputTile[((_ic) * inputTileYSize + (_ity)) * inputTileXSize + (_itx)]

#define FILTER(_oc,_ic,_y,_x) LOAD(filter,(((_oc) * icSize + (_ic)) * fySize + (_y)) * fxSize + (_x))

#define WRITEOUTPUT(_n,_oc,_y,_x,_value) STORE(output,((_n) * ocSize + (_oc)) * xyStride + (_y) * xSize + (_x),_value)
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
  int ntxtySizePadded,
  int xyStride // Stride between spatial planes (may be > xSize*ySize due to padding)
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

#define INPUT(_nic,_xy) LOAD(input,((_nic) * xyStride) + (_xy))
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

string OpenCLKernels::winogradBNActTransformNCHW = OpenCLKernels::common + R"%%(

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

//Activation function
//ACTIVATION (see activations.h)

__kernel void bnActTransform(
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
  int ntxtySizePadded,
  int xyStride // Stride between spatial planes (may be > xSize*ySize due to padding)
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

#define INPUT(_nic,_xy) LOAD(input,((_nic) * xyStride) + (_xy))
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
#if ACTIVATION == 0
        value = (INPUT(nic,xy) * LOAD(scale,ic) + LOAD(bias,ic)) * LOAD(mask, n * xyStride + xy);
#elif ACTIVATION == 1
        value = fmax(INPUT(nic,xy) * LOAD(scale,ic) + LOAD(bias,ic), ZERO) * LOAD(mask, n * xyStride + xy);
#elif ACTIVATION == 2
        float a = INPUT(nic,xy) * LOAD(scale,ic) + LOAD(bias,ic);
        value = floatToReal(a * tanh(a < LOG1PEXPTHRESHOLD ? log1p(exp(a)) : a)) * LOAD(mask, n * xyStride + xy);
#elif ACTIVATION == 12
        float a = INPUT(nic,xy) * LOAD(scale,ic) + LOAD(bias,ic);
        value = floatToReal(a < (LOG1PEXPTHRESHOLD*0.125f) ? a * tanh(log1p(exp(a*8.0f))) : a) * LOAD(mask, n * xyStride + xy);
#elif ACTIVATION == 3
        float a = INPUT(nic,xy) * LOAD(scale,ic) + LOAD(bias,ic);
        value = floatToReal(a / (1.0f + exp(-a))) * LOAD(mask, n * xyStride + xy);
#endif
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
  int ntxtySizePadded,
  int xyStride // Stride between spatial planes (may be > xSize*ySize due to padding)
) {
  const int tileX = get_global_id(0);
  const int tileY = get_global_id(1);
  const int noc = get_global_id(2);
  const int n = noc / ocSize;
  const int oc = noc % ocSize;

  const int ntile = (n * numTilesY + tileY) * numTilesX + tileX;

#define WTILE(_y,_x) wTile[(_y)*INTILE_XSIZE + (_x)]
#define TRANS(_suby,_subx,_oc,_ntile) LOAD(transformed,(((_suby) * INTILE_XSIZE + (_subx))*ocSizePadded + (_oc)) * ntxtySizePadded + (_ntile))
#define WRITEOUTPUT(_noc,_y,_x,_value) STORE(output,(_noc) * xyStride + (_y) * xSize + (_x),_value)

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

string OpenCLKernels::scaleBiasMaskActNCHW = OpenCLKernels::common + R"%%(
__kernel void scaleBiasMaskActNCHW(
  __global realstore* input,  //N, c, H, W
  __global realstore* output, //N, c, H, W, might be the same as input
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
#if ACTIVATION == 0
      real result = (LOAD(input,idx) * LOAD(scale,c) + LOAD(bias,c)) * LOAD(mask,n * xySize + xy);
#elif ACTIVATION == 1
      real result = fmax(LOAD(input,idx) * LOAD(scale,c) + LOAD(bias,c), ZERO) * LOAD(mask,n * xySize + xy);
#elif ACTIVATION == 2
      float a = LOAD(input,idx) * LOAD(scale,c) + LOAD(bias,c);
      real result = floatToReal(a * tanh(a < LOG1PEXPTHRESHOLD ? log1p(exp(a)) : a)) * LOAD(mask,n * xySize + xy);
#elif ACTIVATION == 12
      float a = LOAD(input,idx) * LOAD(scale,c) + LOAD(bias,c);
      real result = floatToReal(a < (LOG1PEXPTHRESHOLD*0.125f) ? a * tanh(log1p(exp(a*8.0f))) : a) * LOAD(mask,n * xySize + xy);
#elif ACTIVATION == 3
      float a = LOAD(input,idx) * LOAD(scale,c) + LOAD(bias,c);
      real result = floatToReal(a / (1.0f + exp(-a))) * LOAD(mask,n * xySize + xy);
#endif
      STORE(output,idx,result);
    }
  }
}
)%%";

string OpenCLKernels::addPointWise = OpenCLKernels::common + R"%%(
#ifndef ELTS_PER_THREAD
  #define ELTS_PER_THREAD 1
#endif
__kernel void addPointWise(
  __global realstore* accum,
  __global realstore* value,
  int size
) {
  const int offsetInTile = get_local_id(0);
  const int tileStart = get_group_id(0) * (get_local_size(0) * ELTS_PER_THREAD);
  #pragma unroll
  for(int d = 0; d < ELTS_PER_THREAD; d++) {
    int s = tileStart + d * get_local_size(0) + offsetInTile;
    if(s < size) {
      real result = LOAD(accum,s) + LOAD(value,s);
      STORE(accum,s,result);
    }
  }
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

string OpenCLKernels::gPoolChannelsNCHWMask = OpenCLKernels::common + R"%%(
//Defines:
//XYSTRIDE - power of two parallelism stride for reduction, should be get_local_size(0)
//CHANNELSTRIDE - stride for channels, should be get_local_size(1)
//LOCALSIZE_TOTAL - should be get_local_size(0) * get_local_size(1) * get_local_size(2)

//PRECONDIION: Kernel is being run where get_num_groups(0) == 1, so that global id and local id are identical for dim 0

__kernel void gPoolChannelsNCHWMask(
  __global realstore* input,  //N, c, HW
  __global float* output, //N, c
  __global realstore* mask,   //N, H, W
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
  float max = -1.0f;
  if(n < nSize && c < cSize) {
    //Sum up the elements that this group member is responsible for
    for(int xy = xyBase; xy < xySize; xy += XYSTRIDE) {
      int idx = (n * cSize + c) * xySize + xy;
      float v = LOAD(input,idx);
      sum += v;
      // Init to -1.0 above and + mask - 1.0 is because it will effectively make all padded space into -1.0
      // which is lower than the lowest value that any current activation function will produce.
      // so the max over all valid spaces will the same as the mask over all spaces including padding.
      // We're relying on all padded space being equal to 0 because this gpool only ever follows a BN+Activate with a mask.
      int maskIdx = n * xySize + xy;
      float maskVal = LOAD(mask,maskIdx);
      max = fmax(max,v + (maskVal-1.0f));
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
#ifndef XY_ELTS_PER_THREAD
  #define XY_ELTS_PER_THREAD 1
#endif
#ifndef NC_ELTS_PER_THREAD
  #define NC_ELTS_PER_THREAD 1
#endif
__kernel void addChannelBiasesNCHW(
  __global realstore* accum,  //NC, HW
  __global float* biases, //NC
  int ncSize,
  int xySize
) {
  const int xyOffsetInTile = get_local_id(0);
  const int xyTileStart = get_group_id(0) * (get_local_size(0) * XY_ELTS_PER_THREAD);
  const int ncBase = get_global_id(1) * NC_ELTS_PER_THREAD;

  #pragma unroll
  for(int r = 0; r < NC_ELTS_PER_THREAD; r++) {
    const int nc = ncBase + r;
    if(nc >= ncSize)
      return;
    real bias = floatToReal(biases[nc]);
    int baseIdx = nc * xySize;
    #pragma unroll
    for(int d = 0; d < XY_ELTS_PER_THREAD; d++) {
      int xy = xyTileStart + d * get_local_size(0) + xyOffsetInTile;
      if(xy < xySize) {
        int idx = baseIdx + xy;
        real result = LOAD(accum,idx) + bias;
        STORE(accum, idx, result);
      }
    }
  }
}
)%%";


string OpenCLKernels::addCBiasesNCAct = OpenCLKernels::common + R"%%(
__kernel void addCBiasesNCAct(
  __global float* accum,  //N,C
  __global float* biases, //C
  int nSize,
  int cSize
) {
  const int c = get_global_id(0);
  const int n = get_global_id(1);

  if(n < nSize && c < cSize) {
#if ACTIVATION == 0
    accum[n * cSize + c] += biases[c];
#elif ACTIVATION == 1
    accum[n * cSize + c] = fmax(accum[n * cSize + c] + biases[c], 0.0f);
#elif ACTIVATION == 2
    float a = accum[n * cSize + c] + biases[c];
    accum[n * cSize + c] = floatToReal(a * tanh(a < LOG1PEXPTHRESHOLD ? log1p(exp(a)) : a));
#elif ACTIVATION == 12
    float a = accum[n * cSize + c] + biases[c];
    accum[n * cSize + c] = floatToReal(a < (LOG1PEXPTHRESHOLD*0.125f) ? a * tanh(log1p(exp(a*8.0f))) : a);
#elif ACTIVATION == 3
    float a = accum[n * cSize + c] + biases[c];
    accum[n * cSize + c] = a / (1.0f + exp(-a));
#endif
  }
}
)%%";


string OpenCLKernels::extractChannel0NCHW = OpenCLKernels::common + R"%%(
__kernel void extractChannel0NCHW(__global realstore* in, __global realstore* out, int nSize, int cSize, int xySize)
{
  const int xyIdx = get_global_id(0);
  const int nIdx = get_global_id(1);
  if(xyIdx < xySize && nIdx < nSize) {
    real result = LOAD(in,nIdx * cSize * xySize + xyIdx);
    STORE(out,nIdx * xySize + xyIdx,result);
  }
}
)%%";

// ============== Transformer Kernels ==============

// Per-channel RMSNorm used inside transformer blocks (weight-only, no bias).
// Input/output are NCHW spatial tensors. Mask is applied.
// Each workgroup of WG_C_SIZE * WG_XY_SIZE threads cooperatively reduces across C channels
// for WG_XY_SIZE spatial positions. Adjacent threads in a warp handle adjacent xy positions
// for the same channel, giving coalesced memory access in NCHW layout.
// Defines: WG_C_SIZE, WG_XY_SIZE - workgroup dimensions (both powers of two)
//          C_PER_THREAD - channels per thread per loop iteration
string OpenCLKernels::transformerRMSNorm = OpenCLKernels::common + R"%%(
__kernel void transformerRMSNorm(
  __global realstore* input,   // N, C, H, W (NCHW)
  __global realstore* output,  // N, C, H, W (NCHW)
  __global float* weight,      // C
  __global realstore* mask,    // N, H, W
  int nSize,
  int cSize,
  int xySize,
  float epsilon
) {
  // Thread decomposition: adjacent threads get adjacent xy for coalesced NCHW access
  const int lid = get_local_id(0);
  const int lid_xy = lid % WG_XY_SIZE;
  const int lid_c = lid / WG_XY_SIZE;
  const int xy = get_group_id(0) * WG_XY_SIZE + lid_xy;
  const int n = get_group_id(1);
  if(n >= nSize) return;

  float maskVal = (xy < xySize) ? LOAD(mask, n * xySize + xy) : 0.0f;

  // Phase 1: Each thread accumulates sum of squares for its channels
  float acc = 0.0f;
  for(int base = lid_c * C_PER_THREAD; base < cSize; base += WG_C_SIZE * C_PER_THREAD) {
    #pragma unroll
    for(int dc = 0; dc < C_PER_THREAD; dc++) {
      int c = base + dc;
      if(c < cSize && xy < xySize) {
        float val = LOAD(input, (n * cSize + c) * xySize + xy) * maskVal;
        acc += val * val;
      }
    }
  }

  // Phase 2: Reduce across C dimension in local memory
  // partials[lid_xy][lid_c] - each xy position reduces independently
  __local float partials[WG_XY_SIZE * WG_C_SIZE];
  partials[lid_xy * WG_C_SIZE + lid_c] = acc;
  for(int span = WG_C_SIZE / 2; span > 0; span /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid_c < span)
      partials[lid_xy * WG_C_SIZE + lid_c] += partials[lid_xy * WG_C_SIZE + lid_c + span];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float rms = rsqrt(partials[lid_xy * WG_C_SIZE] / (float)cSize + epsilon);

  // Phase 3: Apply normalization
  for(int base = lid_c * C_PER_THREAD; base < cSize; base += WG_C_SIZE * C_PER_THREAD) {
    #pragma unroll
    for(int dc = 0; dc < C_PER_THREAD; dc++) {
      int c = base + dc;
      if(c < cSize && xy < xySize) {
        float val = LOAD(input, (n * cSize + c) * xySize + xy);
        float result = val * rms * weight[c] * maskVal;
        STORE(output, (n * cSize + c) * xySize + xy, floatToReal(result));
      }
    }
  }
}
)%%";

// Spatial RMSNorm for trunk tip (rsnh suffix).
// Three-kernel approach for deterministic parallel reduction:
//   Pass 1 (SumSq): Compute masked sum of squares from NCHW input, producing partial sums.
//   Pass 2 (Reduce): Reduce partial sums to a single sum per batch element.
//   Apply: Normalize with gamma, beta, mask using the computed RMS.
//
// The two reduction kernels share the same structure, parametrized by TILE_SIZE.
// Each workgroup handles a contiguous chunk of TILE_SIZE * tilesPerGroup elements.
// Threads accumulate, then reduce in local memory; the leader writes one output.

// Pass 1: Square + mask reduction from NCHW input to partial sums.
// Input: [N, C, xySize] (NCHW), Mask: [N, xySize]
// Output: [N, numWorkgroups] partial sums of squares.
string OpenCLKernels::transformerSpatialRMSNormSumSq = OpenCLKernels::common + R"%%(
//Defines:
//TILE_SIZE - workgroup size (power of two)

__kernel void transformerSpatialRMSNormSumSq(
  __global realstore* input,   // N, C, xySize (NCHW)
  __global realstore* mask,    // N, xySize
  __global float* output,      // N, numCHWWorkgroups (partial sums)
  int nSize,
  int cSize,
  int xySize,
  int tilesPerGroup
) {
  const int lid = get_local_id(0);
  const int n = get_group_id(1);
  if(n >= nSize) return;

  const int groupIdx = get_group_id(0);
  const int totalElems = cSize * xySize;
  const int chunkStart = groupIdx * TILE_SIZE * tilesPerGroup;

  float acc = 0.0f;
  for(int t = 0; t < tilesPerGroup; t++) {
    int idx = chunkStart + t * TILE_SIZE + lid;
    if(idx < totalElems) {
      int c = idx / xySize;
      int xy = idx % xySize;
      float maskVal = LOAD(mask, n * xySize + xy);
      float val = LOAD(input, (n * cSize + c) * xySize + xy) * maskVal;
      acc += val * val;
    }
  }

  __local float partials[TILE_SIZE];
  partials[lid] = acc;
  for(int span = TILE_SIZE / 2; span > 0; span /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid < span)
      partials[lid] += partials[lid + span];
  }

  if(lid == 0) {
    int numWorkgroups = get_num_groups(0);
    output[n * numWorkgroups + groupIdx] = partials[0];
  }
}
)%%";

// Pass 2: Plain sum reduction from partial sums to a single value per batch element.
// Input: [N, numPartials] floats (partial sums from pass 1), Output: [N, 1].
string OpenCLKernels::transformerSpatialRMSNormReduce = R"%%(
//Defines:
//TILE_SIZE - workgroup size (power of two), same as pass 1

__kernel void transformerSpatialRMSNormReduce(
  __global float* input,       // N, numPartials
  __global float* output,      // N, 1
  int nSize,
  int numPartials,
  int tilesPerGroup
) {
  const int lid = get_local_id(0);
  const int n = get_group_id(1);
  if(n >= nSize) return;

  const int groupIdx = get_group_id(0);
  const int chunkStart = groupIdx * TILE_SIZE * tilesPerGroup;

  float acc = 0.0f;
  for(int t = 0; t < tilesPerGroup; t++) {
    int idx = chunkStart + t * TILE_SIZE + lid;
    if(idx < numPartials) {
      acc += input[n * numPartials + idx];
    }
  }

  __local float partials[TILE_SIZE];
  partials[lid] = acc;
  for(int span = TILE_SIZE / 2; span > 0; span /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid < span)
      partials[lid] += partials[lid + span];
  }

  if(lid == 0) {
    int numWorkgroups = get_num_groups(0);
    output[n * numWorkgroups + groupIdx] = partials[0];
  }
}
)%%";

// Apply: Read sum of squares (single value per batch element from reduction),
// compute rms, apply normalization with gamma/beta/mask.
string OpenCLKernels::transformerSpatialRMSNormApply = OpenCLKernels::common + R"%%(
//Defines:
//APPLY_ELTS_PER_THREAD - elements per work-item

__kernel void transformerSpatialRMSNormApply(
  __global realstore* input,   // N, C, xySize (NCHW)
  __global realstore* output,  // N, C, xySize (NCHW)
  __global float* gamma,       // C
  __global float* beta,        // C
  __global realstore* mask,    // N, xySize
  __global float* maskSum,     // N
  __global float* sumSqBuf,    // N (final sum of squares from reduction)
  int nSize,
  int cSize,
  int xySize,
  float epsilon
) {
  const int n = get_group_id(1);
  if(n >= nSize) return;

  float mSum = maskSum[n];
  float denom = mSum * (float)cSize;
  float rms = rsqrt(sumSqBuf[n] / denom + epsilon);

  const int totalElems = cSize * xySize;
  int idx = (int)get_group_id(0) * (int)get_local_size(0) * APPLY_ELTS_PER_THREAD + (int)get_local_id(0);

  for(int e = 0; e < APPLY_ELTS_PER_THREAD; e++) {
    if(idx < totalElems) {
      int c = idx / xySize;
      int xy = idx % xySize;
      float maskVal = LOAD(mask, n * xySize + xy);
      float val = LOAD(input, (n * cSize + c) * xySize + xy);
      float result = (val * rms * gamma[c] + beta[c]) * maskVal;
      STORE(output, (n * cSize + c) * xySize + xy, floatToReal(result));
    }
    idx += (int)get_local_size(0);
  }
}
)%%";

// Apply Rotary Position Embeddings (RoPE) to Q or K tensors.
// Input/output: (N, numHeads, headDim, HW) - after rearranging from NCHW projection
// Cos/sin tables: (HW, numPairs) where numPairs = headDim/2
// For learnable RoPE: cos/sin are precomputed from frequencies per head.
// Rotation: for each pair (x0, x1): out0 = x0*cos - x1*sin, out1 = x0*sin + x1*cos
string OpenCLKernels::transformerApplyRoPE = OpenCLKernels::common + R"%%(
__kernel void transformerApplyRoPE(
  __global realstore* data,     // N, numBufHeads, headDim, HW - modified in place
  __global float* cosTable,     // numKVHeads, numPairs, HW (learnable) or numPairs, HW (fixed)
  __global float* sinTable,     // same layout as cosTable
  int nSize,
  int numBufHeads,
  int numKVHeads,
  int headDim,
  int xySize,
  int numPairs,
  int learnableRope   // 1 = per-head tables, 0 = shared tables
) {
  const int xy = get_global_id(0);
  const int pairIdx = get_global_id(1);  // which pair within headDim (0..numPairs-1)
  const int nh = get_global_id(2);       // n * numBufHeads + h

  int n = nh / numBufHeads;
  int h = nh % numBufHeads;

  if(n < nSize && pairIdx < numPairs && xy < xySize) {
    int idx0 = ((n * numBufHeads + h) * headDim + pairIdx * 2) * xySize + xy;
    int idx1 = ((n * numBufHeads + h) * headDim + pairIdx * 2 + 1) * xySize + xy;

    float x0 = LOAD(data, idx0);
    float x1 = LOAD(data, idx1);

    int tableIdx;
    if(learnableRope) {
      int kvh = h * numKVHeads / numBufHeads;
      tableIdx = (kvh * numPairs + pairIdx) * xySize + xy;
    } else {
      tableIdx = pairIdx * xySize + xy;
    }

    float cosVal = cosTable[tableIdx];
    float sinVal = sinTable[tableIdx];

    float out0 = x0 * cosVal - x1 * sinVal;
    float out1 = x0 * sinVal + x1 * cosVal;

    STORE(data, idx0, floatToReal(out0));
    STORE(data, idx1, floatToReal(out1));
  }
}
)%%";

// FlashAttention-style scaled dot product attention.
// Q: (N*numHeads, headDim, seqLen) - where seqLen = HW
// K: (N*numKVHeads, headDim, seqLen)
// V: (N*numKVHeads, vHeadDim, seqLen)
// mask: (N, seqLen)
// Output: (N*numHeads, vHeadDim, seqLen)
//
// Uses workgroup-tiled approach with online softmax:
// - A workgroup of ATTN_BLOCK_Q work-items handles ATTN_BLOCK_Q query positions for one head.
// - Iterates over key/value positions in tiles of ATTN_BLOCK_KV.
// - K/V tiles are loaded cooperatively into local memory, then each work-item
//   computes its dot products against the shared tile.
//
// Compile-time defines:
// ATTN_BLOCK_Q  - number of threads per workgroup (= local size dim 0)
// ATTN_BLOCK_KV - tile size for key/value iteration
// Q_PER_THREAD  - query positions per thread (workgroup handles ATTN_BLOCK_Q * Q_PER_THREAD total)
// ATTN_HEAD_DIM - the head dimension for Q/K
// ATTN_V_HEAD_DIM - the head dimension for V
string OpenCLKernels::transformerScaledDotProductAttention = R"%%(
#ifndef PRECISION_STORAGE
  #define PRECISION_STORAGE 32
#endif
#if PRECISION_STORAGE == 16
  #pragma OPENCL EXTENSION cl_khr_fp16: enable
  typedef half realstore;
  #define LOAD(__buf,__x) vload_half((__x),(__buf))
  #define STORE(__buf,__x,__y) vstore_half((__y),(__x),(__buf))
#else
  typedef float realstore;
  #define LOAD(__buf,__x) ((__buf)[(__x)])
  #define STORE(__buf,__x,__y) ((__buf)[(__x)] = (__y))
#endif

#ifndef Q_PER_THREAD
  #define Q_PER_THREAD 1
#endif

__kernel void scaledDotProductAttention(
  __global realstore* Q,       // (N*numHeads, headDim, seqLen)
  __global realstore* K,       // (N*numKVHeads, headDim, seqLen)
  __global realstore* V,       // (N*numKVHeads, vHeadDim, seqLen)
  __global realstore* output,  // (N*numHeads, vHeadDim, seqLen)
  __global realstore* mask,    // (N, seqLen) - 0 for masked positions
  int seqLen,
  int numHeads,
  int numKVHeads,
  float scale                  // 1/sqrt(headDim)
) {
  const int localIdx = get_local_id(0);   // 0..ATTN_BLOCK_Q-1
  const int qBlockStart = get_group_id(0) * (ATTN_BLOCK_Q * Q_PER_THREAD);
  const int bh = get_global_id(1);        // batch * numHeads + head

  const int n = bh / numHeads;
  const int h = bh % numHeads;
  const int kvh = h / (numHeads / numKVHeads);
  const int kvBase = n * numKVHeads + kvh;

  // Local memory for K and V tiles
  __local float kTile[ATTN_BLOCK_KV * ATTN_HEAD_DIM];
  __local float vTile[ATTN_BLOCK_KV * ATTN_V_HEAD_DIM];
  __local float kMaskTile[ATTN_BLOCK_KV];

  // Load query vectors and masks for all Q_PER_THREAD positions into private registers
  float q[Q_PER_THREAD * ATTN_HEAD_DIM];
  float qMask[Q_PER_THREAD];
  float runningMax[Q_PER_THREAD];
  float runningSum[Q_PER_THREAD];
  float acc[Q_PER_THREAD * ATTN_V_HEAD_DIM];

  for(int qi = 0; qi < Q_PER_THREAD; qi++) {
    int qPos = qBlockStart + qi * ATTN_BLOCK_Q + localIdx;
    qMask[qi] = 0.0f;
    if(qPos < seqLen) {
      qMask[qi] = LOAD(mask, n * seqLen + qPos);
      if(qMask[qi] != 0.0f) {
        for(int d = 0; d < ATTN_HEAD_DIM; d++) {
          q[qi * ATTN_HEAD_DIM + d] = LOAD(Q, (bh * ATTN_HEAD_DIM + d) * seqLen + qPos);
        }
      }
    }
    runningMax[qi] = -1e30f;
    runningSum[qi] = 0.0f;
    for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
      acc[qi * ATTN_V_HEAD_DIM + d] = 0.0f;
    }
  }

  // Iterate over key/value positions in tiles
  for(int kvStart = 0; kvStart < seqLen; kvStart += ATTN_BLOCK_KV) {
    // Cooperatively load K tile into local memory
    for(int t = localIdx; t < ATTN_BLOCK_KV * ATTN_HEAD_DIM; t += ATTN_BLOCK_Q) {
      int tileKPos = t / ATTN_HEAD_DIM;
      int tileD = t % ATTN_HEAD_DIM;
      int globalKPos = kvStart + tileKPos;
      if(globalKPos < seqLen) {
        kTile[tileKPos * ATTN_HEAD_DIM + tileD] = LOAD(K, (kvBase * ATTN_HEAD_DIM + tileD) * seqLen + globalKPos);
      } else {
        kTile[tileKPos * ATTN_HEAD_DIM + tileD] = 0.0f;
      }
    }

    // Cooperatively load V tile
    for(int t = localIdx; t < ATTN_BLOCK_KV * ATTN_V_HEAD_DIM; t += ATTN_BLOCK_Q) {
      int tileKPos = t / ATTN_V_HEAD_DIM;
      int tileD = t % ATTN_V_HEAD_DIM;
      int globalKPos = kvStart + tileKPos;
      if(globalKPos < seqLen) {
        vTile[tileKPos * ATTN_V_HEAD_DIM + tileD] = LOAD(V, (kvBase * ATTN_V_HEAD_DIM + tileD) * seqLen + globalKPos);
      } else {
        vTile[tileKPos * ATTN_V_HEAD_DIM + tileD] = 0.0f;
      }
    }

    // Cooperatively load mask for this KV tile
    for(int t = localIdx; t < ATTN_BLOCK_KV; t += ATTN_BLOCK_Q) {
      int globalKPos = kvStart + t;
      if(globalKPos < seqLen) {
        kMaskTile[t] = LOAD(mask, n * seqLen + globalKPos);
      } else {
        kMaskTile[t] = 0.0f;
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int kvEnd = min(ATTN_BLOCK_KV, seqLen - kvStart);

    // Each thread processes Q_PER_THREAD query positions against the shared KV tile
    for(int qi = 0; qi < Q_PER_THREAD; qi++) {
      int qPos = qBlockStart + qi * ATTN_BLOCK_Q + localIdx;
      if(qPos < seqLen && qMask[qi] != 0.0f) {
        for(int tileK = 0; tileK < kvEnd; tileK++) {
          if(kMaskTile[tileK] == 0.0f)
            continue;

          // Dot product Q . K
          float dot = 0.0f;
          for(int d = 0; d < ATTN_HEAD_DIM; d++) {
            dot += q[qi * ATTN_HEAD_DIM + d] * kTile[tileK * ATTN_HEAD_DIM + d];
          }
          dot *= scale;

          // Online softmax update
          float newMax = fmax(runningMax[qi], dot);
          float expOldMax = exp(runningMax[qi] - newMax);
          float expCur = exp(dot - newMax);

          for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
            acc[qi * ATTN_V_HEAD_DIM + d] *= expOldMax;
          }
          runningSum[qi] = runningSum[qi] * expOldMax + expCur;
          runningMax[qi] = newMax;

          for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
            acc[qi * ATTN_V_HEAD_DIM + d] += expCur * vTile[tileK * ATTN_V_HEAD_DIM + d];
          }
        }
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write output for all Q_PER_THREAD positions
  for(int qi = 0; qi < Q_PER_THREAD; qi++) {
    int qPos = qBlockStart + qi * ATTN_BLOCK_Q + localIdx;
    if(qPos < seqLen) {
      if(qMask[qi] == 0.0f) {
        for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
          STORE(output, (bh * ATTN_V_HEAD_DIM + d) * seqLen + qPos, 0.0f);
        }
      } else {
        float invSum = (runningSum[qi] > 0.0f) ? (1.0f / runningSum[qi]) : 0.0f;
        for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
          float result = acc[qi * ATTN_V_HEAD_DIM + d] * invSum;
          STORE(output, (bh * ATTN_V_HEAD_DIM + d) * seqLen + qPos, result);
        }
      }
    }
  }
}
)%%";

// Naive (non-tiled) attention kernel: one work-item per (query_position, head).
// Simpler, no shared memory, no barriers. May be faster for small seqLen or small headDim.
// Same interface as the tiled version except no workgroup coordination.
// Compile-time defines: ATTN_HEAD_DIM, ATTN_V_HEAD_DIM
string OpenCLKernels::transformerScaledDotProductAttentionNaive = R"%%(
#ifndef PRECISION_STORAGE
  #define PRECISION_STORAGE 32
#endif
#if PRECISION_STORAGE == 16
  #pragma OPENCL EXTENSION cl_khr_fp16: enable
  typedef half realstore;
  #define LOAD(__buf,__x) vload_half((__x),(__buf))
  #define STORE(__buf,__x,__y) vstore_half((__y),(__x),(__buf))
#else
  typedef float realstore;
  #define LOAD(__buf,__x) ((__buf)[(__x)])
  #define STORE(__buf,__x,__y) ((__buf)[(__x)] = (__y))
#endif

__kernel void scaledDotProductAttentionNaive(
  __global realstore* Q,       // (N*numHeads, headDim, seqLen)
  __global realstore* K,       // (N*numKVHeads, headDim, seqLen)
  __global realstore* V,       // (N*numKVHeads, vHeadDim, seqLen)
  __global realstore* output,  // (N*numHeads, vHeadDim, seqLen)
  __global realstore* mask,    // (N, seqLen) - 0 for masked positions
  int seqLen,
  int numHeads,
  int numKVHeads,
  float scale                  // 1/sqrt(headDim)
) {
  const int qPos = get_global_id(0);
  const int bh = get_global_id(1);  // batch * numHeads + head

  const int n = bh / numHeads;
  const int h = bh % numHeads;
  const int kvh = h / (numHeads / numKVHeads);
  const int kvBase = n * numKVHeads + kvh;

  if(qPos >= seqLen)
    return;

  float qMask = LOAD(mask, n * seqLen + qPos);
  if(qMask == 0.0f) {
    for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
      STORE(output, (bh * ATTN_V_HEAD_DIM + d) * seqLen + qPos, 0.0f);
    }
    return;
  }

  // Load query vector into private registers
  float q[ATTN_HEAD_DIM];
  for(int d = 0; d < ATTN_HEAD_DIM; d++) {
    q[d] = LOAD(Q, (bh * ATTN_HEAD_DIM + d) * seqLen + qPos);
  }

  // Online softmax: iterate over all key positions
  float runningMax = -1e30f;
  float runningSum = 0.0f;
  float acc[ATTN_V_HEAD_DIM];
  for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
    acc[d] = 0.0f;
  }

  for(int kPos = 0; kPos < seqLen; kPos++) {
    float kMask = LOAD(mask, n * seqLen + kPos);
    if(kMask == 0.0f)
      continue;

    // Dot product Q . K
    float dot = 0.0f;
    for(int d = 0; d < ATTN_HEAD_DIM; d++) {
      float kVal = LOAD(K, (kvBase * ATTN_HEAD_DIM + d) * seqLen + kPos);
      dot += q[d] * kVal;
    }
    dot *= scale;

    // Online softmax update
    float newMax = fmax(runningMax, dot);
    float expOldMax = exp(runningMax - newMax);
    float expCur = exp(dot - newMax);

    for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
      acc[d] *= expOldMax;
    }
    runningSum = runningSum * expOldMax + expCur;
    runningMax = newMax;

    for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
      float vVal = LOAD(V, (kvBase * ATTN_V_HEAD_DIM + d) * seqLen + kPos);
      acc[d] += expCur * vVal;
    }
  }

  // Normalize and write output
  float invSum = (runningSum > 0.0f) ? (1.0f / runningSum) : 0.0f;
  for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
    float result = acc[d] * invSum;
    STORE(output, (bh * ATTN_V_HEAD_DIM + d) * seqLen + qPos, result);
  }
}
)%%";

// SwiGLU: output = SiLU(linear1(x)) * linearGate(x)
// Both inputs are already computed.
// SwiGLU activation: output = SiLU(main_proj) * gate_proj
// Flat grid-stride over all N*C*XY elements, no mask needed (inputs already masked by prior layers).
// Defines: ELTS_PER_THREAD - elements per work-item
string OpenCLKernels::transformerSwiGLU = OpenCLKernels::common + R"%%(
__kernel void transformerSwiGLU(
  __global realstore* main_proj,
  __global realstore* gate_proj,
  __global realstore* output,
  int size
) {
  const int tileStart = get_group_id(0) * (int)(get_local_size(0)) * ELTS_PER_THREAD;
  const int lid = get_local_id(0);
  #pragma unroll
  for(int d = 0; d < ELTS_PER_THREAD; d++) {
    int s = tileStart + d * (int)(get_local_size(0)) + lid;
    if(s < size) {
      float a = LOAD(main_proj, s);
      float b = LOAD(gate_proj, s);
      float silu_a = a / (1.0f + exp(-a));
      STORE(output, s, floatToReal(silu_a * b));
    }
  }
}
)%%";

//.
string OpenCLKernels::xgemmDirect =
#include "../external/clblast/common.opencl"
#include "../external/clblast/xgemm_direct_part1.opencl"
#include "../external/clblast/xgemm_direct_part2.opencl"
#include "../external/clblast/xgemm_direct_part3.opencl"
#include "../external/clblast/xgemm_direct_batched.opencl"
;

string OpenCLKernels::xgemm =
"#define ROUTINE_GEMMBATCHED\n"
#include "../external/clblast/common.opencl"
#include "../external/clblast/xgemm_part1a.opencl"
#include "../external/clblast/xgemm_part1b.opencl"
#include "../external/clblast/xgemm_part2.opencl"
#include "../external/clblast/xgemm_part3.opencl"
#include "../external/clblast/xgemm_batched.opencl"
;

string OpenCLKernels::hgemmWmma =
""
#include "../external/clblast/common.opencl"
#include "../neuralnet/hgemm_wmma.opencl"
;

string OpenCLKernels::hgemmWmmaNCHW =
""
#include "../external/clblast/common.opencl"
#include "../neuralnet/hgemm_wmma_nchw.opencl"
#include "../neuralnet/hgemm_wmma_nchw_part2.opencl"
;

#endif
