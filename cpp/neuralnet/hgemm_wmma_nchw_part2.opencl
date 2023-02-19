/*
 * Second part of hgemm_wmma_nchw.opencl
 * Author: David J Wu ("lightvector")
 */
R"%%(
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
    LocalToGlobalCComplete((__global short*)(cgm + ((groupId1 * NWG)*hwSize)), clm, tid, groupId0 * MWG, hwSize, numThreads);
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
    LocalToGlobalCEdge((__global short*)(cgm + ((groupId1 * NWG)*hwSize)), clm, tid, groupId0 * MWG, hwSize, numThreads);
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
