/*
 * This is an FP16 tensor-core implementation of matrix multiplication, designed to work with
 * in conjunction with the same "common.opencl" of the CLBlast library's xgemm, and heavily
 * based around similar structures as their implementation and shares the vast majority of their
 * naming convention for things. It is also very similar to an implementation by Leela Zero.
 * See also https://github.com/CNugteren/CLBlast and https://github.com/leela-zero/leela-zero
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
#ifndef VWM
  //Vector width for loading data
  #define VWM 1
#endif
#ifndef VWN
  //Vector width for loading data
  #define VWN 1
#endif
#ifndef SA
  //Use local memory?
  #define SA 0
#endif
#ifndef SB
  //Use local memory?
  #define SB 0
#endif

#define KDIM_WMMA 16
#define WARP_SIZE 32

//Number of waves handled by a workgroup
#define MWI (MWG/MWAVE)
#define NWI (NWG/NWAVE)

#if SA == 1
  #if MWARP == 16 && NWARP == 16
  #define WMMA_LOAD_A "wmma.load.a.sync.aligned.col.m16n16k16.shared.f16"
  #elif MWARP == 8 && NWARP == 32
  #define WMMA_LOAD_A "wmma.load.a.sync.aligned.col.m8n32k16.shared.f16"
  #elif MWARP == 32 && NWARP == 8
  #define WMMA_LOAD_A "wmma.load.a.sync.aligned.col.m32n8k16.shared.f16"
  #endif
#else
  #if MWARP == 16 && NWARP == 16
  #define WMMA_LOAD_A "wmma.load.a.sync.aligned.col.m16n16k16.global.f16"
  #elif MWARP == 8 && NWARP == 32
  #define WMMA_LOAD_A "wmma.load.a.sync.aligned.col.m8n32k16.global.f16"
  #elif MWARP == 32 && NWARP == 8
  #define WMMA_LOAD_A "wmma.load.a.sync.aligned.col.m32n8k16.global.f16"
  #endif
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
#define WMMA_STORE "wmma.store.d.sync.aligned.col.m16n16k16.global.f16"
#elif MWARP == 8 && NWARP == 32
#define WMMA_MMA "wmma.mma.sync.aligned.col.row.m8n32k16.f16.f16"
#define WMMA_STORE "wmma.store.d.sync.aligned.col.m8n32k16.global.f16"
#elif MWARP == 32 && NWARP == 8
#define WMMA_MMA "wmma.mma.sync.aligned.col.row.m32n8k16.f16.f16"
#define WMMA_STORE "wmma.store.d.sync.aligned.col.m32n8k16.global.f16"
#endif


#if VWM == 1
  #define LOADM(__buf,__x) vload((__x),(__buf))
  #define STOREM(__buf,__x,__val) vstore((__val),(__x),(__buf))
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
  #define LOADN(__buf,__x) vload((__x),(__buf))
  #define STOREN(__buf,__x,__val) vstore((__val),(__x),(__buf))
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


#if SA == 1
//Loads a MWG * KWG sized chunk from agm to alm
INLINE_FUNC void GlobalToLocalA(
  const __global short* restrict agm, LOCAL_PTR short* alm,
  const int tid, const int kSizeM, const int numThreads
) {
  const int tileSizeInVecs = MWG * KWG / VWM;
  const int srcStrideInVecs = kSizeM / VWM;
  #pragma unroll
  for(int i = tid; i < tileSizeInVecs; i += numThreads) {
    int m = i % (MWG / VWM);
    int k = i / (MWG / VWM);
    int srcIdx = m + k*srcStrideInVecs;
    int dstIdx = i;
    STOREM(alm, dstIdx, LOADM(agm, srcIdx));
  }
}
#endif

#if SB == 1
//Loads a NWG * KWG sized chunk from bgm to blm
INLINE_FUNC void GlobalToLocalB(
  const __global short* restrict bgm, LOCAL_PTR short* blm,
  const int tid, const int kSizeN, const int numThreads
) {
  const int tileSizeInVecs = NWG * KWG / VWN;
  const int srcStrideInVecs = kSizeN / VWN;
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

INLINE_FUNC void hGemmWmmaBody(
  const int kSizeM, const int kSizeN, const int kSizeK,
  const __global half* restrict agm, const __global half* restrict bgm,
  __global half* cgm
  #if SA == 1 && SB == 1
  , LOCAL_PTR short* alm, LOCAL_PTR short* blm
  #elif SA == 1
  , LOCAL_PTR short* alm
  #elif SB == 1
  , LOCAL_PTR short* blm
  #endif
) {
  //printf("OVERALL %d %d %d %p %p %p %p %p %p\n", kSizeM, kSizeN, kSizeK, agm, agm+kSizeM*kSizeK, bgm, bgm+kSizeN*kSizeK, cgm, cgm+kSizeM*kSizeN);
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

  //Process KWG-size chunks of K at a time.
  for(int kwg = 0; kwg < kSizeK; kwg += KWG) {
    #if SA == 1
      GlobalToLocalA((const __global short*)(agm + (kwg*kSizeM + groupId0 * MWG)), alm, tid, kSizeM, numThreads);
    #endif
    #if SB == 1
      GlobalToLocalB((const __global short*)(bgm + (kwg*kSizeN + groupId1 * NWG)), blm, tid, kSizeN, numThreads);
    #endif
    #if SA == 1 || SB == 1
      barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    //Process KDIM_WMMA-size chunks of the KWG-sized chunk of K.
    //KDIM_WMMA is the size of
    for(int kWaveOffset = 0; kWaveOffset<KWG; kWaveOffset += KDIM_WMMA) {

      //Process MWAVE-sized chunks of MWG at a time, with each local_id within our group handling an MWARP chunk.
      //Preload nto registers to cut in half the number of loads.
      #pragma promote_to_registers
      int a[MWI][8];
      for(int aWaveId = 0; aWaveId<MWI; aWaveId++) {
        const int aOffset = aWaveId * MWAVE + wmmaBlockId0 * MWARP;
        #if SA == 1
          const int aStride = MWG;
          const LOCAL_PTR half* aSrc = (const LOCAL_PTR half*)(alm + (aOffset + kWaveOffset*aStride));
        #else
          const int aStride = kSizeM;
          const __global half* aSrc = agm + (aOffset + groupId0*MWG + (kWaveOffset+kwg)*aStride);
        #endif

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
          const int bStride = kSizeN;
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
    const int bOffset = bWaveId * NWAVE + wmmaBlockId1 * NWARP + groupId1 * NWG;
    #pragma unroll
    for(int aWaveId = 0; aWaveId<MWI; aWaveId++) {
      const int aOffset = aWaveId * MWAVE + wmmaBlockId0 * MWARP + groupId0 * MWG;
      const int c0 = c0Acc[bWaveId][aWaveId];
      const int c1 = c1Acc[bWaveId][aWaveId];
      const int c2 = c2Acc[bWaveId][aWaveId];
      const int c3 = c3Acc[bWaveId][aWaveId];
      __global half* dst = cgm + (kSizeM * bOffset + aOffset);
      asm("{" WMMA_STORE " [%0], {%1,%2,%3,%4}, %5;}" :
        :
        "l"(dst), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(kSizeM)
      );
    }
  }
}

__kernel __attribute__((reqd_work_group_size(MWAVE/MWARP*WARP_SIZE, NWAVE/NWARP, 1)))
void hgemmWmmaBatched(
  const int kSizeM, const int kSizeN, const int kSizeK,
  const __global half* restrict agm,
  const __global half* restrict bgm,
  __global half* restrict cgm
  ) {
  const int batch = get_group_id(2);
  const int a_offset = batch * kSizeM * kSizeK;
  const int b_offset = batch * kSizeN * kSizeK;
  const int c_offset = batch * kSizeM * kSizeN;
  const __global half* restrict agm_ = &agm[a_offset];
  const __global half* restrict bgm_ = &bgm[b_offset];
  __global half* restrict cgm_ = &cgm[c_offset];

  #if SA == 1
    __local short alm[KWG * MWG] __attribute__ ((aligned (32)));
  #endif
  #if SB == 1
    __local short blm[KWG * NWG] __attribute__ ((aligned (32)));
  #endif

  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    hGemmWmmaBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alm, blm);
  #elif SA == 1
    hGemmWmmaBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alm);
  #elif SB == 1
    hGemmWmmaBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, blm);
  #else
    hGemmWmmaBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_);
  #endif
}

)%%"
