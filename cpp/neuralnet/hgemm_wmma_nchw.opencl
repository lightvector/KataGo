/*
 * This is similar to hgemm_wmma.opencl, but designed to assume matrix A is in NCHW format
 * and automatically adds appropriate padding when loading matrix A.
 * Requires and assumes that C is a multiple of KWG, and that matrix B's dimensions are a
 * multiple of NWG and KWG.
 * Assumes matrix B has C as the outer dimension and OC as the inner dimension.
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


// A is shape [C,H,W], row major
// B is shape [C,OC], row major
// Relative to hgemm_wmma.opencl, "hwSize" is "kSizeM" and "cSize" is kSizeK" and "ocSize" is "kSizeN".
INLINE_FUNC void hGemmWmmaCHWBody(
  const int cSize, const int hwSize, const int ocSize,
  const __global half* restrict agm, const __global half* restrict bgm,
  __global half* cgm
  #if SB == 1
  , LOCAL_PTR short* alm, LOCAL_PTR short* blm, LOCAL_PTR short* clm
  #else
  , LOCAL_PTR short* alm, LOCAL_PTR short* clm
  #endif
) {
  const int groupId0 = get_group_id(0);
  const int groupId1 = get_group_id(1);
  const int wmmaBlockId0 = get_local_id(0) / WARP_SIZE;
  const int wmmaBlockId1 = get_local_id(1);
  const int tid = get_local_id(0) + get_local_id(1) * get_local_size(0);
  const int numThreads = get_local_size(0) * get_local_size(1);

  #pragma promote_to_registers
  int c0Acc[NWI][MWI];
  #pragma promote_to_registers
  int c1Acc[NWI][MWI];
  #pragma promote_to_registers
  int c2Acc[NWI][MWI];
  #pragma promote_to_registers
  int c3Acc[NWI][MWI];
  #pragma unroll
  for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
    #pragma unroll
    for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
      c0Acc[bWaveId][aWaveId] = 0;
      c1Acc[bWaveId][aWaveId] = 0;
      c2Acc[bWaveId][aWaveId] = 0;
      c3Acc[bWaveId][aWaveId] = 0;
    }
  }

  // Complete tiles
  if((groupId0+1) * MWG <= hwSize) {
    //Process KWG-size chunks of cSize at a time.
    for(int kwg = 0; kwg < cSize; kwg += KWG) {
      GlobalToLocalAComplete((const __global short*)(agm + (kwg*hwSize)), alm, tid, groupId0 * MWG, hwSize, numThreads);
      #if SB == 1
        GlobalToLocalB((const __global short*)(bgm + (kwg*ocSize + groupId1 * NWG)), blm, tid, ocSize, numThreads);
      #endif

      barrier(CLK_LOCAL_MEM_FENCE);

      //Process KDIM_WMMA-size chunks of the KWG-sized chunk of K.
      //KDIM_WMMA is the size of
      for(int kWaveOffset = 0; kWaveOffset<KWG; kWaveOffset += KDIM_WMMA) {

        //Process MWAVE-sized chunks of MWG at a time, with each local_id within our group handling an MWARP chunk.
        //Preload into registers to cut in half the number of loads.
        #pragma promote_to_registers
        int a[MWI][8];
        for(int aWaveId = 0; aWaveId<MWI; aWaveId++) {
          const int aOffset = aWaveId * MWAVE + wmmaBlockId0 * MWARP;

          const int aStride = MWG;
          const LOCAL_PTR half* aSrc = (const LOCAL_PTR half*)(alm + (aOffset + kWaveOffset*aStride));

          /*
          printf("ReadingA tid %d (%d,%d) awid %d wmmab %d / kwg %d kwoff %d %p %d %d\n", tid, groupId0, groupId1, aWaveId, wmmaBlockId0, kwg, kWaveOffset, aSrc,
                  aOffset + groupId0*MWG + (kWaveOffset+kwg)*aStride, aStride);
          */
          asm("{" WMMA_LOAD_A " {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;}" :
            "=r"(a[aWaveId][0]),
            "=r"(a[aWaveId][1]),
            "=r"(a[aWaveId][2]),
            "=r"(a[aWaveId][3]),
            "=r"(a[aWaveId][4]),
            "=r"(a[aWaveId][5]),
            "=r"(a[aWaveId][6]),
            "=r"(a[aWaveId][7])
            :
            "l"(aSrc),
            "r"(aStride)
          );
        }

        //Process NWAVE-sized chunks of NWG at a time, with each local_id within our group handling an NWARP chunk.
        for(int bWaveId = 0; bWaveId<NWI; bWaveId++) {
          const int bOffset = bWaveId * NWAVE + wmmaBlockId1 * NWARP;
          #if SB == 1
            const int bStride = NWG;
            const LOCAL_PTR half* bSrc = (const LOCAL_PTR half*)(blm + (bOffset + kWaveOffset*bStride));
          #else
            const int bStride = ocSize;
            const __global half* bSrc = bgm + (bOffset + groupId1*NWG + (kWaveOffset+kwg)*bStride);
          #endif

          /*
          printf("ReadingB %d (%d,%d) %d %d / %d %d %p %d %d\n", tid, groupId0, groupId1, bWaveId, wmmaBlockId1, kwg, kWaveOffset, bSrc,
                  bOffset + groupId1*NWG + (kWaveOffset+kwg)*bStride, bStride);
          */
          #pragma promote_to_registers
          int b[8];
          asm("{" WMMA_LOAD_B " {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;}" :
            "=r"(b[0]),
            "=r"(b[1]),
            "=r"(b[2]),
            "=r"(b[3]),
            "=r"(b[4]),
            "=r"(b[5]),
            "=r"(b[6]),
            "=r"(b[7])
            :
            "l"(bSrc),
            "r"(bStride)
          );

          //Iterate back over our stored A's and perform our matrix mult
          for(int aWaveId = 0; aWaveId<MWI; aWaveId++) {
            int r0,r1,r2,r3;
            asm("{" WMMA_MMA " {%0,%1,%2,%3}, {%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19}, {%20,%21,%22,%23};}" :
              "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
              :
              "r"(a[aWaveId][0]), "r"(a[aWaveId][1]), "r"(a[aWaveId][2]), "r"(a[aWaveId][3]), "r"(a[aWaveId][4]), "r"(a[aWaveId][5]), "r"(a[aWaveId][6]), "r"(a[aWaveId][7]),
              "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]),
              "r"(c0Acc[bWaveId][aWaveId]), "r"(c1Acc[bWaveId][aWaveId]), "r"(c2Acc[bWaveId][aWaveId]), "r"(c3Acc[bWaveId][aWaveId])
            );
            c0Acc[bWaveId][aWaveId] = r0;
            c1Acc[bWaveId][aWaveId] = r1;
            c2Acc[bWaveId][aWaveId] = r2;
            c3Acc[bWaveId][aWaveId] = r3;
          }
        } //Close bWaveId loop

      } //Close kWaveOffset
    } //Close kwg loop

    #pragma unroll
    for(int bWaveId = 0; bWaveId<NWI; bWaveId++) {
      const int bOffset = bWaveId * NWAVE + wmmaBlockId1 * NWARP; // + groupId1 * NWG; this part we don't need because we're storing to clm instead of cgm
      #pragma unroll
      for(int aWaveId = 0; aWaveId<MWI; aWaveId++) {
        const int aOffset = aWaveId * MWAVE + wmmaBlockId0 * MWARP; // + groupId0 * MWG; this part we don't need because we're storing to clm instead of cgm
        const int c0 = c0Acc[bWaveId][aWaveId];
        const int c1 = c1Acc[bWaveId][aWaveId];
        const int c2 = c2Acc[bWaveId][aWaveId];
        const int c3 = c3Acc[bWaveId][aWaveId];
        //__global half* dst = cgm + (kSizeM * bOffset + aOffset);
        const int mStride = MWG;
        LOCAL_PTR half* dst = (LOCAL_PTR half*)(clm + (mStride * bOffset + aOffset));
        asm("{" WMMA_STORE " [%0], {%1,%2,%3,%4}, %5;}" :
          :
          "l"(dst), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mStride)
        );
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    LocalToGlobalCComplete((const __global short*)(cgm + ((groupId1 * NWG)*hwSize)), clm, tid, groupId0 * MWG, hwSize, numThreads);
  }
  else {
    //Process KWG-size chunks of cSize at a time.
    for(int kwg = 0; kwg < cSize; kwg += KWG) {
      GlobalToLocalAEdge((const __global short*)(agm + (kwg*hwSize)), alm, tid, groupId0 * MWG, hwSize, numThreads);
      #if SB == 1
        GlobalToLocalB((const __global short*)(bgm + (kwg*ocSize + groupId1 * NWG)), blm, tid, ocSize, numThreads);
      #endif

      barrier(CLK_LOCAL_MEM_FENCE);

      //Process KDIM_WMMA-size chunks of the KWG-sized chunk of K.
      //KDIM_WMMA is the size of
      for(int kWaveOffset = 0; kWaveOffset<KWG; kWaveOffset += KDIM_WMMA) {

        //Process MWAVE-sized chunks of MWG at a time, with each local_id within our group handling an MWARP chunk.
        //Preload into registers to cut in half the number of loads.
        #pragma promote_to_registers
        int a[MWI][8];
        for(int aWaveId = 0; aWaveId<MWI; aWaveId++) {
          const int aOffset = aWaveId * MWAVE + wmmaBlockId0 * MWARP;

          const int aStride = MWG;
          const LOCAL_PTR half* aSrc = (const LOCAL_PTR half*)(alm + (aOffset + kWaveOffset*aStride));

          /*
          printf("ReadingA tid %d (%d,%d) awid %d wmmab %d / kwg %d kwoff %d %p %d %d\n", tid, groupId0, groupId1, aWaveId, wmmaBlockId0, kwg, kWaveOffset, aSrc,
                  aOffset + groupId0*MWG + (kWaveOffset+kwg)*aStride, aStride);
          */
          asm("{" WMMA_LOAD_A " {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;}" :
            "=r"(a[aWaveId][0]),
            "=r"(a[aWaveId][1]),
            "=r"(a[aWaveId][2]),
            "=r"(a[aWaveId][3]),
            "=r"(a[aWaveId][4]),
            "=r"(a[aWaveId][5]),
            "=r"(a[aWaveId][6]),
            "=r"(a[aWaveId][7])
            :
            "l"(aSrc),
            "r"(aStride)
          );
        }

        //Process NWAVE-sized chunks of NWG at a time, with each local_id within our group handling an NWARP chunk.
        for(int bWaveId = 0; bWaveId<NWI; bWaveId++) {
          const int bOffset = bWaveId * NWAVE + wmmaBlockId1 * NWARP;
          #if SB == 1
            const int bStride = NWG;
            const LOCAL_PTR half* bSrc = (const LOCAL_PTR half*)(blm + (bOffset + kWaveOffset*bStride));
          #else
            const int bStride = ocSize;
            const __global half* bSrc = bgm + (bOffset + groupId1*NWG + (kWaveOffset+kwg)*bStride);
          #endif

          /*
          printf("ReadingB %d (%d,%d) %d %d / %d %d %p %d %d\n", tid, groupId0, groupId1, bWaveId, wmmaBlockId1, kwg, kWaveOffset, bSrc,
                  bOffset + groupId1*NWG + (kWaveOffset+kwg)*bStride, bStride);
          */
          #pragma promote_to_registers
          int b[8];
          asm("{" WMMA_LOAD_B " {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;}" :
            "=r"(b[0]),
            "=r"(b[1]),
            "=r"(b[2]),
            "=r"(b[3]),
            "=r"(b[4]),
            "=r"(b[5]),
            "=r"(b[6]),
            "=r"(b[7])
            :
            "l"(bSrc),
            "r"(bStride)
          );

          //Iterate back over our stored A's and perform our matrix mult
          for(int aWaveId = 0; aWaveId<MWI; aWaveId++) {
            int r0,r1,r2,r3;
            asm("{" WMMA_MMA " {%0,%1,%2,%3}, {%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19}, {%20,%21,%22,%23};}" :
              "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
              :
              "r"(a[aWaveId][0]), "r"(a[aWaveId][1]), "r"(a[aWaveId][2]), "r"(a[aWaveId][3]), "r"(a[aWaveId][4]), "r"(a[aWaveId][5]), "r"(a[aWaveId][6]), "r"(a[aWaveId][7]),
              "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]),
              "r"(c0Acc[bWaveId][aWaveId]), "r"(c1Acc[bWaveId][aWaveId]), "r"(c2Acc[bWaveId][aWaveId]), "r"(c3Acc[bWaveId][aWaveId])
            );
            c0Acc[bWaveId][aWaveId] = r0;
            c1Acc[bWaveId][aWaveId] = r1;
            c2Acc[bWaveId][aWaveId] = r2;
            c3Acc[bWaveId][aWaveId] = r3;
          }
        } //Close bWaveId loop

      } //Close kWaveOffset
    } //Close kwg loop

    #pragma unroll
    for(int bWaveId = 0; bWaveId<NWI; bWaveId++) {
      const int bOffset = bWaveId * NWAVE + wmmaBlockId1 * NWARP; // + groupId1 * NWG; this part we don't need because we're storing to clm instead of cgm
      #pragma unroll
      for(int aWaveId = 0; aWaveId<MWI; aWaveId++) {
        const int aOffset = aWaveId * MWAVE + wmmaBlockId0 * MWARP; // + groupId0 * MWG; this part we don't need because we're storing to clm instead of cgm
        const int c0 = c0Acc[bWaveId][aWaveId];
        const int c1 = c1Acc[bWaveId][aWaveId];
        const int c2 = c2Acc[bWaveId][aWaveId];
        const int c3 = c3Acc[bWaveId][aWaveId];
        //__global half* dst = cgm + (kSizeM * bOffset + aOffset);
        const int mStride = MWG;
        LOCAL_PTR half* dst = (LOCAL_PTR half*)(clm + (mStride * bOffset + aOffset));
        asm("{" WMMA_STORE " [%0], {%1,%2,%3,%4}, %5;}" :
          :
          "l"(dst), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mStride)
        );
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    LocalToGlobalCEdge((const __global short*)(cgm + ((groupId1 * NWG)*hwSize)), clm, tid, groupId0 * MWG, hwSize, numThreads);
  }
}

__kernel __attribute__((reqd_work_group_size(MWAVE/MWARP*WARP_SIZE, NWAVE/NWARP, 1)))
void hgemmWmmaNCHW(
  const int cSize, const int hwSize, const int ocSize,
  const __global half* restrict agm,
  const __global half* restrict bgm,
  __global half* restrict cgm
  ) {
  const int batch = get_group_id(2);
  const int a_offset = batch * cSize * hwSize;
  const int c_offset = batch * ocSize * hwSize;
  const __global half* restrict agm_ = &agm[a_offset];
  const __global half* restrict bgm_ = bgm;
  __global half* restrict cgm_ = &cgm[c_offset];

  __local short alm[KWG * MWG] __attribute__ ((aligned (32)));
  #if SB == 1
    __local short blm[KWG * NWG] __attribute__ ((aligned (32)));
  #endif
  __local short clm[NWG * MWG] __attribute__ ((aligned (32)));

  // Computes the matrix-multiplication and stores the result in global memory
  #if SB == 1
    hGemmWmmaCHWBody(cSize, hwSize, ocSize, agm_, bgm_, cgm_, alm, blm, clm);
  #else
    hGemmWmmaCHWBody(cSize, hwSize, ocSize, agm_, bgm_, cgm_, alm, clm);
  #endif
}

)%%"
