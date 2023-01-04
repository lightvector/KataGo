/*
 * This is similar to hgemm_wmma.opencl, but designed to assume matrix A is in NCHW format
 * and automatically adds appropriate padding when loading matrix A.
 * Requires and assumes that C is a multiple of KWG, and that matrix B's dimensions are a
 * multiple of NWG and KWG.
 * Assumes matrix B has C as the outer dimension and OC as the inner dimension.
 * The second part of this implemenation is hgemm_wmma_nchw_part2.opencl
 *
 * Author: David J Wu ("lightvector")
 */
R"%%(

//External parameters
#ifndef MWG
  //Size in m dimension of the tile that every local workgroup is responsible for
  #define MWG 32
#endif
#ifndef NWG
  //Size in n dimension of the tile that every local workgroup is responsible for
  #define NWG 32
#endif
#ifndef KWG
  //Size in k dimension of the tile that we load into local memory at a time.
  //Each workgroup processes the *entire* k dimension at a time, in chunks of this size.
  #define KWG 32
#endif
#ifndef MWAVE
  //Total amount of M that we process in parallel across an entire workgroup per wave of wmma calls
  //by all the warps in a workgroup
  #define MWAVE 32
#endif
#ifndef NWAVE
  //Total amount of N that we process in parallel across an entire workgroup per wave of wmma calls
  //by all the warps in a workgroup
  #define NWAVE 32
#endif
#ifndef MWARP
  //Total amount of M that we process per wmma call, by a single warp
  #define MWARP 16
#endif
#ifndef NWARP
  //Total amount of N that we process per wmma call, by a single warp
  #define NWARP 16
#endif

//Vector width for loading data
#ifndef VWM
  //Vector width for loading data
  #define VWM 1
#endif

#ifndef VWN
  //Vector width for loading data
  #define VWN 1
#endif

//SA is always 1 for this implementation because it also performs padding to handle the NCHW
//This define is not actually used, we specialized the implementation to assume it is 1.
#define SA 1

#ifndef SB
  //Use local memory?
  #define SB 0
#endif

#define KDIM_WMMA 16
#define WARP_SIZE 32

//Number of waves handled by a workgroup
#define MWI (MWG/MWAVE)
#define NWI (NWG/NWAVE)

#if MWARP == 16 && NWARP == 16
#define WMMA_LOAD_A "wmma.load.a.sync.aligned.col.m16n16k16.shared.f16"
#elif MWARP == 8 && NWARP == 32
#define WMMA_LOAD_A "wmma.load.a.sync.aligned.col.m8n32k16.shared.f16"
#elif MWARP == 32 && NWARP == 8
#define WMMA_LOAD_A "wmma.load.a.sync.aligned.col.m32n8k16.shared.f16"
#endif

#if SB == 1
  #if MWARP == 16 && NWARP == 16
  #define WMMA_LOAD_B "wmma.load.b.sync.aligned.row.m16n16k16.shared.f16"
  #elif MWARP == 8 && NWARP == 32
  #define WMMA_LOAD_B "wmma.load.b.sync.aligned.row.m8n32k16.shared.f16"
  #elif MWARP == 32 && NWARP == 8
  #define WMMA_LOAD_B "wmma.load.b.sync.aligned.row.m32n8k16.shared.f16"
  #endif
#else
  #if MWARP == 16 && NWARP == 16
  #define WMMA_LOAD_B "wmma.load.b.sync.aligned.row.m16n16k16.global.f16"
  #elif MWARP == 8 && NWARP == 32
  #define WMMA_LOAD_B "wmma.load.b.sync.aligned.row.m8n32k16.global.f16"
  #elif MWARP == 32 && NWARP == 8
  #define WMMA_LOAD_B "wmma.load.b.sync.aligned.row.m32n8k16.global.f16"
  #endif
#endif

#if MWARP == 16 && NWARP == 16
#define WMMA_MMA "wmma.mma.sync.aligned.col.row.m16n16k16.f16.f16"
#define WMMA_STORE "wmma.store.d.sync.aligned.col.m16n16k16.shared.f16"
#elif MWARP == 8 && NWARP == 32
#define WMMA_MMA "wmma.mma.sync.aligned.col.row.m8n32k16.f16.f16"
#define WMMA_STORE "wmma.store.d.sync.aligned.col.m8n32k16.shared.f16"
#elif MWARP == 32 && NWARP == 8
#define WMMA_MMA "wmma.mma.sync.aligned.col.row.m32n8k16.f16.f16"
#define WMMA_STORE "wmma.store.d.sync.aligned.col.m32n8k16.shared.f16"
#endif

#define LOAD1M(__buf,__x) ((__buf)[(__x)])
#define STORE1M(__buf,__x,__val) ((__buf)[(__x)] = (__val))

#if VWM == 1
  #define LOADM(__buf,__x) ((__buf)[(__x)])
  #define STOREM(__buf,__x,__val) ((__buf)[(__x)] = (__val))
#elif VWM == 2
  #define LOADM(__buf,__x) vload2((__x),(__buf))
  #define STOREM(__buf,__x,__val) vstore2((__val),(__x),(__buf))
#elif VWM == 4
  #define LOADM(__buf,__x) vload4((__x),(__buf))
  #define STOREM(__buf,__x,__val) vstore4((__val),(__x),(__buf))
#elif VWM == 8
  #define LOADM(__buf,__x) vload8((__x),(__buf))
  #define STOREM(__buf,__x,__val) vstore8((__val),(__x),(__buf))
#elif VWM == 16
  #define LOADM(__buf,__x) vload16((__x),(__buf))
  #define STOREM(__buf,__x,__val) vstore16((__val),(__x),(__buf))
#endif

#if VWN == 1
  #define LOADN(__buf,__x) ((__buf)[(__x)])
  #define STOREN(__buf,__x,__val) ((__buf)[(__x)] = (__val))
#elif VWN == 2
  #define LOADN(__buf,__x) vload2((__x),(__buf))
  #define STOREN(__buf,__x,__val) vstore2((__val),(__x),(__buf))
#elif VWN == 4
  #define LOADN(__buf,__x) vload4((__x),(__buf))
  #define STOREN(__buf,__x,__val) vstore4((__val),(__x),(__buf))
#elif VWN == 8
  #define LOADN(__buf,__x) vload8((__x),(__buf))
  #define STOREN(__buf,__x,__val) vstore8((__val),(__x),(__buf))
#elif VWN == 16
  #define LOADN(__buf,__x) vload16((__x),(__buf))
  #define STOREN(__buf,__x,__val) vstore16((__val),(__x),(__buf))
#endif


//Loads a KWG * MWG sized chunk from agm to alm, padding with zeros if hw goes out of bounds.
//K is the outer dimension
//Unlike GlobalToLocalB, expects agm to point to the start of the hw dimension, rather than be pre-offset for the chunk
//to be loaded, and makes up for it by passing the offset, hwStart, here.
INLINE_FUNC void GlobalToLocalAEdge(
  const __global short* restrict agm, LOCAL_PTR short* alm,
  const int tid, const int hwStart, const int hwSize, const int numThreads
) {
  const int tileSize = MWG * KWG;
  const int srcStride = hwSize;
  #pragma unroll
  for(int i = tid; i < tileSize; i += numThreads) {
    int m = i % MWG;
    int k = i / MWG;
    int hw = m+hwStart;

    short val = 0;
    int dstIdx = i;
    if(hw < hwSize) {
      int srcIdx = hw + k*srcStride;
      val = LOAD1M(agm, srcIdx);
    }
    STORE1M(alm, dstIdx, val);
  }
}

//Handles complete tiles, does not check hwSize, vectorized transfer
INLINE_FUNC void GlobalToLocalAComplete(
  const __global short* restrict agm, LOCAL_PTR short* alm,
  const int tid, const int hwStart, const int hwSize, const int numThreads
) {
  const int tileSizeInVecs = MWG * KWG / VWM;
  #pragma unroll
  for(int i = tid; i < tileSizeInVecs; i += numThreads) {
    int m = i % (MWG / VWM);
    int k = i / (MWG / VWM);

    int dstIdx = i;
    int srcIdx = m;
    STOREM(alm, dstIdx, LOADM((agm+hwStart+k*hwSize), srcIdx));
  }
}

#if SB == 1
//Loads a KWG * NWG sized chunk from bgm to blm
//K is the outer dimension
INLINE_FUNC void GlobalToLocalB(
  const __global short* restrict bgm, LOCAL_PTR short* blm,
  const int tid, const int ocSize, const int numThreads
) {
  const int tileSizeInVecs = NWG * KWG / VWN;
  const int srcStrideInVecs = ocSize / VWN;
  #pragma unroll
  for(int i = tid; i < tileSizeInVecs; i += numThreads) {
    int n = i % (NWG / VWN);
    int k = i / (NWG / VWN);
    int srcIdx = n + k*srcStrideInVecs;
    int dstIdx = i;
    STOREN(blm, dstIdx, LOADN(bgm, srcIdx));
  }
}
#endif


//Stores a NWG * MWG sized chunk from clm to cgm, skipping where hw goes out of bounds.
//N is the outer dimension.
//NWG goes across oc, MWG goes across hw.
//Unlike GlobalToLocalB, expects cgm to point to the start of the hw dimension, rather than be pre-offset for the chunk
//to be loaded, and makes up for it by passing the offset, hwStart, here.
INLINE_FUNC void LocalToGlobalCEdge(
  __global short* restrict cgm, const LOCAL_PTR short* clm,
  const int tid, const int hwStart, const int hwSize, const int numThreads
) {
  const int tileSize = MWG * NWG;
  const int dstStride = hwSize;
  #pragma unroll
  for(int i = tid; i < tileSize; i += numThreads) {
    int m = i % MWG;
    int n = i / MWG;
    int hw = m+hwStart;

    if(hw < hwSize) {
      int dstIdx = hw + n*dstStride;
      int srcIdx = i;
      STORE1M(cgm, dstIdx, LOAD1M(clm, srcIdx));
    }
  }
}

//Handles complete tiles, does not check hwSize, vectorized transfer
INLINE_FUNC void LocalToGlobalCComplete(
  __global short* restrict cgm, const LOCAL_PTR short* clm,
  const int tid, const int hwStart, const int hwSize, const int numThreads
) {
  const int tileSizeInVecs = MWG * NWG / VWM;
  #pragma unroll
  for(int i = tid; i < tileSizeInVecs; i += numThreads) {
    int m = i % (MWG / VWM);
    int n = i / (MWG / VWM);

    int dstIdx = m;
    int srcIdx = i;
    STOREM((cgm+hwStart+n*hwSize), dstIdx, LOADM(clm, srcIdx));
  }
}

)%%"
