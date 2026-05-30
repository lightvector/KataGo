// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverlength-strings"
#endif
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4068)
#endif
namespace cudnn_frontend {
namespace experimental {
namespace generated {
inline constexpr const char sm100_d64_fprop_source[] =
    R"KERNEL(

// receive_op 0 includes
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef long long int64_t;
typedef int int32_t;
typedef short int16_t;
typedef signed char int8_t;
typedef unsigned int r32;

struct CUtensorMap {
  alignas(64) uint64_t opaque[16];
};

#define __FLT_MIN__ 1.17549435082228750796873653722224568e-38F
#define __FLT_MAX__ 3.40282346638528859811704183484516925e+38F

#define CUDACC_VERSION __CUDACC_VER_MAJOR__ * 10 + __CUDACC_VER_MINOR__

namespace fort {
static const uint64_t AMPERE_MEM_DESC_DEFAULT = uint64_t(0x1000000000000000ul);
static const uint64_t MEM_DESC_DEFAULT = AMPERE_MEM_DESC_DEFAULT;
static const uint32_t MMA_PEER_BIT_MASK = 0xFEFFFFFF;

#define NAN __int_as_float(0x7fffffff)
#define POS_INFINITY __int_as_float(0x7f800000)
#define NEG_INFINITY __int_as_float(0xff800000)

typedef uint16_t bfloat16_t;

#define FORT_MIN(a, b) ((a) < (b) ? (a) : (b))
#define FORT_MAX(a, b) ((a) > (b) ? (a) : (b))
#define FORT_DIV_UP(a, b) (((a) + (b) - 1) / (b))
#define FORT_ROUND_UP(a, b) ((((a) + (b) - 1) / (b)) * (b))

typedef struct tensor_descriptor {
  static const int MAX_DIMS = 12;

  int64_t num_dims;
  int64_t dims[MAX_DIMS];
  int64_t strides[MAX_DIMS];
} tensor_descriptor;

typedef struct FastDivisor {
  uint32_t val, shr, mul;
} FastDivisor_t;

inline __device__ void fastDivMod(const FastDivisor_t &d, uint32_t val,
                                  uint32_t &div, uint32_t &mod) {
  div = __umulhi((uint32_t)2 * val, d.mul) >> d.shr;
  mod = val - div * d.val;
}

inline __device__ char *get_smem_loc_epilogue_swizzle_128b(
    char *smem_addr, int local_block_id, int tid, int local_row, int column,
    size_t element_size, int block_size, int row_per_tile) {
  if (element_size * block_size == 256) {
    // Case 1: 2 * swizzle tile size = block size -> half block per swizzle tile
    int swizzle_col = local_row ^ (column % 8);

    // (local_block_id * 2 + (column / 8)) * row_per_tile * 128: gives offset of
    // corresponding swizzle tile tid * 128: tid gives number of loads in block
    // swizzle_col * 16B: swizzle_col gives transformed column
    return smem_addr +
           (local_block_id * 2 + (column / 8)) * row_per_tile * 128 +
           tid * 128 + swizzle_col * 16;
  } else if (element_size * block_size == 128) {
    // Case 2: swizzle tile size = block size -> one block per swizzle tile
    int swizzle_col = local_row ^ (column % 8);

    // local_block_id * row_per_tile * 128: gives offset of corresponding
    // swizzle tile tid * 128: tid gives number of loads in block swizzle_col *
    // 16B: swizzle_col gives transformed column
    return smem_addr + local_block_id * row_per_tile * 128 + tid * 128 +
           swizzle_col * 16;
  } else {
    // Case 3: swizzle tile size / 2 = block size -> two blocks per swizzle tile
    int swizzle_col = local_row ^ ((column + local_block_id * 4) % 8);

    // (local_block_id / 2) * row_per_tile * 128: gives offset of corresponding
    // swizzle tile tid * 128: tid gives number of loads in block swizzle_col *
    // 16B: swizzle_col gives transformed column
    return smem_addr + (local_block_id / 2) * row_per_tile * 128 + tid * 128 +
           swizzle_col * 16;
  }
}

inline __device__ static void get_next_block_id(uint32_t result,
                                                uint32_t mbar) {
#if __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
  asm volatile("{clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::"
               "complete_tx::bytes.b128 [%0], [%1];}\n\t" ::"r"(result),
               "r"(mbar));
#else
  asm volatile(
      "{clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_"
      "tx::bytes.multicast::cluster::all.b128 [%0], [%1];}\n\t" ::"r"(result),
      "r"(mbar));
#endif
}

inline __device__ float2 ffma2(const float2 &a, const float2 &b,
                               const float2 &c) {
  uint64_t d;
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
               : "=l"(d)
               : "l"(reinterpret_cast<const uint64_t &>(a)),
                 "l"(reinterpret_cast<const uint64_t &>(b)),
                 "l"(reinterpret_cast<const uint64_t &>(c)));
  return reinterpret_cast<float2 &>(d);
}

inline __device__ float2 fadd2(const float2 &a, const float2 &b) {
  uint64_t c;
  asm volatile("add.f32x2 %0, %1, %2;\n"
               : "=l"(c)
               : "l"(reinterpret_cast<const uint64_t &>(a)),
                 "l"(reinterpret_cast<const uint64_t &>(b)));
  return reinterpret_cast<float2 &>(c);
}

inline __device__ float2 fmul2(const float2 &a, const float2 &b) {
  uint64_t c;
  asm volatile("mul.f32x2 %0, %1, %2;\n"
               : "=l"(c)
               : "l"(reinterpret_cast<const uint64_t &>(a)),
                 "l"(reinterpret_cast<const uint64_t &>(b)));
  return reinterpret_cast<float2 &>(c);
} 

inline __device__ float fmax3(float a, float b, float c) {
#if (__CUDACC_VER_MAJOR__ >= 13)
  float d;
  asm volatile("max.ftz.f32 %0, %1, %2, %3;\n"
               : "=f"(d)
               : "f"((a)), "f"((b)), "f"((c)));
  return d;
#else
  #error "This kernel is not supported for CUDA toolkit version < 13.0"
  return 0.0f;
#endif
}

inline __device__ float row_max_reduction_128_elems(uint32_t reg[128]) {
  float tmp_max_0[42];
  float tmp_max_1[14];
  float tmp_max_2[5];
  float tmp_max_3[2];

#pragma unroll
  for (int i = 0; i < 126; i += 3) {
    tmp_max_0[i / 3] = fmax3(reinterpret_cast<float &>(reg[i + 0]),
                             reinterpret_cast<float &>(reg[i + 1]),
                             reinterpret_cast<float &>(reg[i + 2]));
  }
#pragma unroll
  for (int i = 0; i < 42; i += 3) {
    tmp_max_1[i / 3] =
        fmax3(tmp_max_0[i + 0], tmp_max_0[i + 1], tmp_max_0[i + 2]);
  }
#pragma unroll
  for (int i = 0; i < 12; i += 3) {
    tmp_max_2[i / 3] =
        fmax3(tmp_max_1[i + 0], tmp_max_1[i + 1], tmp_max_1[i + 2]);
  }
  tmp_max_2[4] =
      fmax3(tmp_max_1[12], tmp_max_1[13], reinterpret_cast<float &>(reg[126]));
  tmp_max_3[0] = fmax3(tmp_max_2[0], tmp_max_2[1], tmp_max_2[2]);
  tmp_max_3[1] =
      fmax3(tmp_max_2[3], tmp_max_2[4], reinterpret_cast<float &>(reg[127]));

  return fmax3(tmp_max_3[0], tmp_max_3[1], tmp_max_3[1]);
}

#if __CUDA_ARCH__ >= 900

inline __device__ uint32_t elect_one_sync() {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile("{\n"
               ".reg .b32 %%rx;\n"
               ".reg .pred %%px;\n"
               "     elect.sync %%rx|%%px, %2;\n"
               "@%%px mov.s32 %1, 1;\n"
               "     mov.s32 %0, %%rx;\n"
               "}\n"
               : "+r"(laneid), "+r"(pred)
               : "r"(0xFFFFFFFF));
  return pred;
}

template <int TARGET_REG_COUNT> inline __device__ void reg_alloc() {
  // const int TARGET_REG_COUNT = 232; // Example values (use with reg_delloc's
  // VALUE together): 224, 232, 208, 216
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n"
               :
               : "n"(TARGET_REG_COUNT));
}

template <int TARGET_REG_COUNT> inline __device__ void reg_dealloc() {
  // const int TARGET_REG_COUNT = 40; // Exmaple values: 56, 40, 88, 72
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n"
               :
               : "n"(TARGET_REG_COUNT));
}

inline __device__ void named_barrier_arrive(uint32_t BARRIER_ID,
                                            uint32_t NUM_THREADS) {
  asm volatile("bar.arrive %0, %1;" : : "r"(BARRIER_ID), "r"(NUM_THREADS));
}

inline __device__ void named_barrier_wait(uint32_t BARRIER_ID,
                                          uint32_t NUM_THREADS) {
  asm volatile("bar.sync %0, %1;" ::"r"(BARRIER_ID), "r"(NUM_THREADS));
}

inline __device__ void tmastg_arrive() {
  asm volatile("cp.async.bulk.commit_group;");
}

template <int Count> inline __device__ void tmastg_wait_count() {
  asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(Count) : "memory");
}

#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
inline __device__ void smem_bar_init(uint32_t smem_ptr, uint32_t thread_count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_ptr),
               "r"(thread_count));
}

inline __device__ void
smem_bar_set_transaction_count(uint32_t smem_ptr, uint32_t expected_copy_bytes,
                               uint32_t pred = 0) {
  asm volatile("{\n\t.reg .pred p;"
               " \n\tsetp.eq.u32 p, %2, 1;"
               " \n\t@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               " \n\t}" ::"r"(smem_ptr),
               "r"(expected_copy_bytes), "r"(pred));
}

inline __device__ uint32_t smem_bar_peek(uint32_t smem_ptr,
                                         uint32_t bar_phase) {
  uint32_t bar_phase_out;
  asm volatile("{\n\t.reg .pred       P1;"
               " \n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;"
               " \n\tselp.b32 %0, 1, 0, P1; \n\t}"
               : "=r"(bar_phase_out)
               : "r"(smem_ptr), "r"(bar_phase));
  return bar_phase_out;
}

inline __device__ void
wait_barrier(uint32_t smem_int_ptr,
             int phase_bit) // Current phase bit the barrier waiting to flip
{
  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@P1                       bra.uni DONE;\n"
               "bra.uni                   LAB_WAIT;\n"
               "DONE:\n"
               "}\n" ::"r"(smem_int_ptr),
               "r"(phase_bit));
}

// Barrier arrive on local smem
inline __device__ void arrive_barrier(uint32_t smem_addr) {
  asm volatile("{\n\t"
               "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"
               "}"
               :
               : "r"(smem_addr));
}

#endif

// Circular Buffer Index + Associated Phase
// Assumes only one operation possible - i.e., ++
template <uint32_t Stages_> struct PipelineState {
public:
  static constexpr int Stages = Stages_;

  inline __device__ PipelineState(int index, int state)
      : state_(state), index_(index) {}

  inline __device__ int phase() const { return state_; }

  inline __device__ int index() const { return index_; }

  inline __device__ void operator++() {
    if (Stages > 1) {
      ++index_;
      if (index_ == Stages) {
        index_ = 0;
        state_ ^= 1;
      }
    } else {
      state_ ^= 1;
    }
  }

  int state_;
  int index_;
};

template <uint64_t MAX_SHIFT_FOR_WS, uint64_t UTCMMA_M, uint64_t UTCMMA_N,
          uint64_t B_TRANSPOSE, uint64_t A_TRANSPOSE, uint64_t B_NEGATE,
          uint64_t A_NEGATE, uint64_t SRC_B_TYPE, uint64_t SRC_A_TYPE,
          uint64_t SPARSE_METADATA_FORMAT, uint64_t ACC_TYPE,
          uint64_t SATURATE_ENABLED, uint64_t SPARSE_ENABLED,
          uint64_t SPARSE_METADATA_ID2>
inline __device__ constexpr uint64_t build_utcmma_instruction_desc() {
  return (static_cast<uint64_t>(0) | (MAX_SHIFT_FOR_WS << 30) |
          ((UTCMMA_M >> 5) << 25) | ((UTCMMA_N >> 3) << 17) |
          (B_TRANSPOSE << 16) | (A_TRANSPOSE << 15) | (B_NEGATE << 14) |
          (A_NEGATE << 13) | (SRC_B_TYPE << 10) | (SRC_A_TYPE << 7) |
          (SPARSE_METADATA_FORMAT << 6) | (ACC_TYPE << 4) |
          (SATURATE_ENABLED << 3) | (SPARSE_ENABLED << 2) |
          (SPARSE_METADATA_ID2 << 0))
         << 32;
}
)KERNEL"
    R"KERNEL(

class Smem_utcmma_descriptor {
public:
  inline __device__ Smem_utcmma_descriptor(uint64_t SWIZZLE_MODE,
                                           uint64_t BASE_OFFSET,
                                           uint64_t DESC_VERSION,
                                           uint64_t BYTES_PER_LEADING_DIM,
                                           uint64_t BYTES_PER_STRIDE_DIM) {
    // ------------------------------------
    // Setup smem descriptor for operand A:
    // ------------------------------------
    // Note 1: SWIZZLE_NONE = 0, SWIZZLE_128B = 2, SWIZZLE_64B = 4, SWIZZLE_32B
    // = 6, SWIZZLE_128B_ATOM32B = 1, N/A = 3, N/A = 5, N/A = 7
    const uint64_t SWIZZLE_MODE_IN_BIT_LOCATION = SWIZZLE_MODE
                                                  << 61; // bits: 63-61

    // Note 2: Base offset. Valid only for matrix descriptor 1, 2 or 3, 4
    const uint64_t BASE_OFFSET_IN_BIT_LOCATION = BASE_OFFSET
                                                 << 49; // bits 51-49

    // Note 3: Descriptor version, needs to be set to 1. ???
    const uint64_t DESC_VERSION_IN_BIT_LOCATION = DESC_VERSION
                                                  << 46; // bits 48-46

    // Note 4: Stride dimension byte offset, 16 byte aligned, 4 LSBs not
    // included
    const uint64_t STRIDE_DIM_BYTE_OFFSET = BYTES_PER_STRIDE_DIM >> 4;
    const uint64_t STRIDE_DIM_BYTE_OFFSET_IN_BIT_LOCATION =
        STRIDE_DIM_BYTE_OFFSET << 32; // bits 45-32

    // Note 5: Leading dimension byte offset, 16 byte aligned, 4 LSBs not
    // included
    const uint64_t LEADING_DIM_BYTE_OFFSET = BYTES_PER_LEADING_DIM >> 4;
    const uint64_t LEADING_DIM_BYTE_OFFSET_IN_BIT_LOCATION =
        LEADING_DIM_BYTE_OFFSET << 16; // bits 29-16

    desc =
        (SWIZZLE_MODE_IN_BIT_LOCATION | BASE_OFFSET_IN_BIT_LOCATION |
         DESC_VERSION_IN_BIT_LOCATION | STRIDE_DIM_BYTE_OFFSET_IN_BIT_LOCATION |
         LEADING_DIM_BYTE_OFFSET_IN_BIT_LOCATION);
  }
  template <int BYTES_PER_BUFFER, int BUFFER_COUNT>
  inline __device__ void set_smem(uint32_t smem) {
    int2 &tmp = reinterpret_cast<int2 &>(desc);
    tmp.x |= (static_cast<uint64_t>(smem & 0xFFFFFF) >> 4);

    int2 &tmp_initial = reinterpret_cast<int2 &>(initial_desc);
    tmp_initial.x = tmp.x;

    max_desc = tmp.x + (BYTES_PER_BUFFER >> 4) * (BUFFER_COUNT - 1);
  }

  template <int BYTES_PER_BUFFER, int BUFFER_COUNT>
  inline __device__ void increment_smem_buffer() {
    int2 &tmp = reinterpret_cast<int2 &>(desc);
    tmp.x += (tmp.x >= max_desc) ? -(BYTES_PER_BUFFER >> 4) * (BUFFER_COUNT - 1)
                                 : (BYTES_PER_BUFFER >> 4);
  }

  template <int BYTES_PER_BUFFER>
  inline __device__ void stage_increment_smem_buffer(int stage) {
    int2 &tmp_initial = reinterpret_cast<int2 &>(initial_desc);
    int2 &tmp = reinterpret_cast<int2 &>(desc);

    tmp.x = tmp_initial.x + (BYTES_PER_BUFFER >> 4) * stage;
  }

  template <int BYTES_OFFSET> inline __device__ void add_smem_offset() {
    int2 &tmp = reinterpret_cast<int2 &>(desc);
    tmp.x += (BYTES_OFFSET >> 4);
  }

  uint64_t initial_desc;
  uint64_t desc;
  uint32_t max_desc;
};

// Adapted from :
// cta1 utcmma with A and B from smem and C stored in tmem, scale denotes if we
// want to accumulate C or not Something like: Accumulate or overwrite C.   1:
// read C, 0: ignore C [clear accumulators]
inline __device__ void
utcmma_asmem_bsmem_h_cta1(uint64_t sdescA, uint64_t sdescB, uint32_t tmemC,
                          uint64_t idescE, uint32_t scaleC) {
  uint32_t mask[4] = {0, 0, 0, 0}; // TODO : Figure out what this mask means,
                                   // CUTLASS hardcodes to all zeros
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "setp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, "
               "%7, %8}, p; \n\t"
               "}\n"
               :
               : "r"(tmemC), "l"(sdescA), "l"(sdescB),
                 "r"(uint32_t(idescE >> 32)), "r"(scaleC), "r"(mask[0]),
                 "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
}

// cta1 utcmma with A from tmem and B from smem and C stored in tmem, scale
// denotes if we want to accumulate C or not Something like: Accumulate or
// overwrite C.   1: read C, 0: ignore C [clear accumulators]
inline __device__ void
utcmma_atmem_bsmem_h_cta1(uint32_t tmemA, uint64_t sdescB, uint32_t tmemC,
                          uint64_t idescE, uint32_t scaleC) {
  uint32_t mask[4] = {0, 0, 0, 0}; // TODO : Figure out what this mask means,
                                   // CUTLASS hardcodes to all zeros
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "setp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, "
               "%6, %7, %8}, p; \n\t"
               "}\n"
               :
               : "r"(tmemC), "r"(tmemA), "l"(sdescB),
                 "r"(uint32_t(idescE >> 32)), "r"(scaleC), "r"(mask[0]),
                 "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
}

// Check for the utcmma to finish
inline __device__ void umma_arrive(uint32_t bar_intptr) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
               "cluster.b64 [%0];" ::"r"(bar_intptr));
}

inline __device__ uint32_t cast_smem_ptr_to_uint(void const *const ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

inline __device__ void fence_view_async_shared(void) {
  asm volatile("fence.proxy.async.shared::cta;\n");
}

inline __device__ void sts_128(r32 dst, r32 src[4]) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(dst),
               "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]));
}

inline __device__ void lds_128(r32 dst[4], r32 src) {
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
               : "r"(src));
}

inline __device__ void lds_32(r32 dst[1], r32 src) {
  asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(dst[0]) : "r"(src));
}

inline __device__ void stg_32(void *ptr, r32 val[1]) {
  uint32_t *p = reinterpret_cast<uint32_t *>(ptr);
  p[0] = val[0];
}

inline __device__ void stg_8(void *ptr, r32 val[1]) {
  uint8_t *p = reinterpret_cast<uint8_t *>(ptr);
  p[0] = reinterpret_cast<const uint8_t &>(val[0]);
}

inline __device__ void ldg_16(r32 dst[1], const void *ptr, bool pred = true) {
  if (pred) {
    *reinterpret_cast<uint16_t *>(dst) =
        *reinterpret_cast<const uint16_t *>(ptr);
  } else {
    *reinterpret_cast<uint16_t *>(dst) = 0;
  }
}

inline __device__ void ldg_8(r32 dst[1], const void *ptr, bool pred = true) {
  if (pred) {
    *reinterpret_cast<uint8_t *>(dst) = *reinterpret_cast<const uint8_t *>(ptr);
  } else {
    *reinterpret_cast<uint8_t *>(dst) = 0;
  }
}

inline __device__ void ldg_16_reg(r32 dst[1], const void *ptr,
                                  bool pred = true) {
  if (pred) {
    uint16_t tmp = *reinterpret_cast<const uint16_t *>(ptr);
    asm volatile("{\nmov.b32 %0, {%1, %1};\n}\n" : "=r"(dst[0]) : "h"(tmp));
  } else {
    dst[0] = 0;
  }
}

inline __device__ void ldg_8_reg(r32 dst[1], const void *ptr,
                                 bool pred = true) {
  if (pred) {
    uint8_t tmp_val = *reinterpret_cast<const uint8_t *>(ptr);
    uint8_t *tmp_ptr = reinterpret_cast<uint8_t *>(dst);
    tmp_ptr[0] = tmp_ptr[1] = tmp_ptr[2] = tmp_ptr[3] = tmp_val;
  } else {
    dst[0] = 0;
  }
}

// Forward (non-ReLU) Activations
// Backward (non-ReLU) Activations
// Forward ReLU activations
// Relu helper function: clip the lower part
// Relu helper function: clip the upper part
// Forward ReLU: clip only lower part
// Forward ReLU: clip lower part and upper part
// Forward ReLU: clip only lower part, but add slope to lower clip
// Forward ReLU: clip lower part and upper part, but add slope to lower clip
#if __CUDA_ARCH__ >= 900
#endif

// optimization opportunity exists here
inline __device__ void fp32x2_to_fp16x2(r32 dst[1], const r32 src[2]) {
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n"
               : "=r"(dst[0])
               : "r"(src[1]), "r"(src[0]));
}

inline __device__ void fp32x2_to_fp16x2_relu(r32 dst[1], const r32 src[2]) {
  asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n"
               : "=r"(dst[0])
               : "r"(src[1]), "r"(src[0]));
}

inline __device__ void fp32x2_to_bf16x2(r32 dst[1], const r32 src[2]) {
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n"
               : "=r"(dst[0])
               : "r"(src[1]), "r"(src[0]));
}

// mma_pipeline_op 1 includes
typedef struct AttentionDescriptor {
  // Input parameters
  // b - batch
  // q_h - num heads of q/dq/o/do
  // k_h - num heads of k/dk
  // v_h - num heads of v/dv
  // s_q - max sequence length of q
  // s_kv - max sequence length of kv
  // d - hidden dim (head dim)
  uint32_t b, q_h, k_h, v_h, s_q, s_kv, d_qk, d_v;
  uint16_t q_heads_per_k, q_heads_per_v, min_q_heads_per_kv;
} AttentionDescriptor_t;


inline __device__ void fence_view_async_tmem_load() {
  asm volatile("{\n\ttcgen05.wait::ld.sync.aligned; \n}" ::);
}

inline __device__ void fence_view_async_tmem_store() {
  asm volatile("{\n\ttcgen05.wait::st.sync.aligned; \n}" ::);
}

inline __device__ void tmem_allocate_1sm(uint32_t num_columns,
                                         uint32_t smem_addr) {
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
          smem_addr),
      "r"(num_columns));
}

inline __device__ void tmem_free_1sm(uint32_t num_columns, uint32_t tmem_addr) {
  asm volatile(
      "{\n\ttcgen05.dealloc.cta_group::1.sync.aligned.b32  %0, %1; \n\t}" ::"r"(
          tmem_addr),
      "r"(num_columns));
}

inline __device__ void tmem_release_allocation_lock_1sm() {
)KERNEL"
    R"KERNEL(
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::);
}

/////////////////////////////////
/// TMEM STTM related instructions
/////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 128 times
// 32 data path lanes, 32-bit pattern, repeated 64 times
// 32 data path lanes, 32-bit pattern, repeated 32 times
inline __device__ void sttm_32dp32bit_x32(uint32_t dst, r32 src[32]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32"
               "[%0],"
               "{%1, %2, %3, %4,"
               "%5, %6, %7, %8,"
               "%9, %10, %11, %12,"
               "%13, %14, %15, %16,"
               "%17, %18, %19, %20,"
               "%21, %22, %23, %24,"
               "%25, %26, %27, %28,"
               "%29, %30, %31, %32};\n"
               :
               : "r"(dst), "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
                 "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]),
                 "r"(src[8]), "r"(src[9]), "r"(src[10]), "r"(src[11]),
                 "r"(src[12]), "r"(src[13]), "r"(src[14]), "r"(src[15]),
                 "r"(src[16]), "r"(src[17]), "r"(src[18]), "r"(src[19]),
                 "r"(src[20]), "r"(src[21]), "r"(src[22]), "r"(src[23]),
                 "r"(src[24]), "r"(src[25]), "r"(src[26]), "r"(src[27]),
                 "r"(src[28]), "r"(src[29]), "r"(src[30]), "r"(src[31]));
}

// 32 data path lanes, 32-bit pattern, repeated 16 times
inline __device__ void sttm_32dp32bit_x16(uint32_t dst, r32 src[16]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32"
               "[%0],"
               "{%1, %2, %3, %4,"
               "%5, %6, %7, %8,"
               "%9, %10, %11, %12,"
               "%13, %14, %15, %16};\n"
               :
               : "r"(dst), "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
                 "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]),
                 "r"(src[8]), "r"(src[9]), "r"(src[10]), "r"(src[11]),
                 "r"(src[12]), "r"(src[13]), "r"(src[14]), "r"(src[15]));
}

// 32 data path lanes, 32-bit pattern, repeated 8 times
inline __device__ void sttm_32dp32bit_x8(uint32_t dst, r32 src[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32"
               "[%0],"
               "{%1, %2, %3, %4,"
               "%5, %6, %7, %8};\n"
               :
               : "r"(dst), "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
                 "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]));
}

// 32 data path lanes, 32-bit pattern, repeated 4 times
// 32 data path lanes, 32-bit pattern, repeated 2 times
inline __device__ void sttm_32dp32bit_x2(uint32_t dst, r32 src[2]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32"
               "[%0],"
               "{%1, %2};\n"
               :
               : "r"(dst), "r"(src[0]), "r"(src[1]));
}

// 32 data path lanes, 32-bit pattern, repeated 1 times
inline __device__ void sttm_32dp32bit_x1(uint32_t dst, r32 src[1]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32"
               "[%0],"
               "{%1};\n"
               :
               : "r"(dst), "r"(src[0]));
}

/////////////////////////////////
/// TMEM LDTM related instructions
/////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 128 times
// 32 data path lanes, 32-bit pattern, repeated 64 times
// 32 data path lanes, 32-bit pattern, repeated 32 times
inline __device__ void ldtm_32dp32bit_x32(r32 dst[32], uint32_t src_addr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32"
               "{%0, %1, %2, %3,"
               "%4, %5, %6, %7,"
               "%8, %9, %10, %11,"
               "%12, %13, %14, %15,"
               "%16, %17, %18, %19,"
               "%20, %21, %22, %23,"
               "%24, %25, %26, %27,"
               "%28, %29, %30, %31},"
               "[%32];\n"
               : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]),
                 "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7]),
                 "=r"(dst[8]), "=r"(dst[9]), "=r"(dst[10]), "=r"(dst[11]),
                 "=r"(dst[12]), "=r"(dst[13]), "=r"(dst[14]), "=r"(dst[15]),
                 "=r"(dst[16]), "=r"(dst[17]), "=r"(dst[18]), "=r"(dst[19]),
                 "=r"(dst[20]), "=r"(dst[21]), "=r"(dst[22]), "=r"(dst[23]),
                 "=r"(dst[24]), "=r"(dst[25]), "=r"(dst[26]), "=r"(dst[27]),
                 "=r"(dst[28]), "=r"(dst[29]), "=r"(dst[30]), "=r"(dst[31])
               : "r"(src_addr));
}

// 32 data path lanes, 32-bit pattern, repeated 16 times
// 32 data path lanes, 32-bit pattern, repeated 8 times
inline __device__ void ldtm_32dp32bit_x8(r32 dst[8], uint32_t src_addr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32"
               "{%0, %1, %2, %3,"
               "%4, %5, %6, %7},"
               "[%8];\n"
               : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]),
                 "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7])
               : "r"(src_addr));
}

// 32 data path lanes, 32-bit pattern, repeated 4 times
// 32 data path lanes, 32-bit pattern, repeated 2 times
inline __device__ void ldtm_32dp32bit_x2(r32 dst[2], uint32_t src_addr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32"
               "{%0, %1},"
               "[%2];\n"
               : "=r"(dst[0]), "=r"(dst[1])
               : "r"(src_addr));
}

// 32 data path lanes, 32-bit pattern, repeated 1 times
inline __device__ void ldtm_32dp32bit_x1(r32 dst[1], uint32_t src_addr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32"
               "{%0},"
               "[%1];\n"
               : "=r"(dst[0])
               : "r"(src_addr));
}

// CUtensorMap, when issued using bulk copy async functions, are cached in
// constant cache. But the producer kernel may have directly written to global
// memory, without invalidating this constant cache. This acquire fence,
// invalidates the memory address in constant cache, using UTMACCTL.IV.
inline __device__ void tma_descriptor_fence_acquire(CUtensorMap const *p_desc) {
#if (__CUDA_ARCH__ >= 900) && (CUDACC_VERSION >= 123)
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(p_desc);
  asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n"
               :
               : "l"(gmem_int_desc)
               : "memory");
  asm volatile("cvta.global.u64 %0, %0;\n"
               :
               : "l"(gmem_int_desc), "l"(gmem_int_desc)
               : "memory");
#endif
}

inline __device__ void
utmaldg_4d_tiled(const void *p_desc,
                 uint32_t urb0, // smem offset
                 uint32_t urb1, // smem barrier offset
                 int32_t urb2,  // m
                 int32_t urb3,  // n
                 int32_t urb4,  // b
                 int32_t urb5,  // extra coord
                 const uint32_t elect_one,
                 const uint64_t mem_desc = MEM_DESC_DEFAULT) {
  if (elect_one) {
#if __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::"
        "bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5}], [%6], %7;\n" ::"r"(
            urb0),
        "l"(reinterpret_cast<uint64_t>(p_desc)), "r"(urb2), "r"(urb3),
        "r"(urb4), "r"(urb5), "r"(urb1), "l"(mem_desc)
        : "memory");
#else
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::"
        "bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5}], [%6], %7;\n" ::"r"(
            urb0),
        "l"(reinterpret_cast<uint64_t>(p_desc)), "r"(urb2), "r"(urb3),
        "r"(urb4), "r"(urb5), "r"(urb1), "l"(mem_desc)
        : "memory");
#endif
  }
}

inline __device__ void
utmaldg_4d_tiled_multicast(const void *p_desc,
                           uint32_t urb0, // smem offset
                           uint32_t urb1, // smem barrier offset
                           int32_t urb2,  // m
                           int32_t urb3,  // n
                           int32_t urb4,  // b
                           int32_t urb5,  // extra coord
                           const uint16_t mcast_mask, const uint32_t elect_one,
                           const uint64_t mem_desc = MEM_DESC_DEFAULT) {
  if (elect_one) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::"
                 "complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], "
                 "[%1, {%2, %3, %4, %5}], [%6], %7, %8;\n" ::"r"(urb0),
                 "l"(reinterpret_cast<uint64_t>(p_desc)), "r"(urb2), "r"(urb3),
                 "r"(urb4), "r"(urb5), "r"(urb1), "h"(mcast_mask), "l"(mem_desc)
                 : "memory");
  }
}

#if (__CUDA_ARCH__ >= 1000) && (CUDACC_VERSION >= 128)
inline __device__ void
utmaldg_4d_tiled_2cta(const void *p_desc,
                      uint32_t urb0, // smem offset
                      uint32_t urb1, // smem barrier offset
                      int32_t urb2,  // m
                      int32_t urb3,  // n
                      int32_t urb4,  // b
                      int32_t urb5,  // extra coord
                      const uint32_t elect_one,
                      const uint64_t mem_desc = MEM_DESC_DEFAULT) {
  if (elect_one) {
    asm volatile("cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global."
                 "mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, "
                 "%3, %4, %5}], [%6], %7;\n" ::"r"(urb0),
                 "l"(reinterpret_cast<uint64_t>(p_desc)), "r"(urb2), "r"(urb3),
                 "r"(urb4), "r"(urb5), "r"(urb1 & MMA_PEER_BIT_MASK),
                 "l"(mem_desc)
                 : "memory");
  }
}

inline __device__ void
utmaldg_4d_tiled_multicast_2cta(const void *p_desc,
                                uint32_t urb0, // smem offset
                                uint32_t urb1, // smem barrier offset
                                int32_t urb2,  // m
                                int32_t urb3,  // n
                                int32_t urb4,  // b
                                int32_t urb5,  // extra coord
                                const uint16_t mcast_mask,
                                const uint32_t elect_one,
                                const uint64_t mem_desc = MEM_DESC_DEFAULT) {
  if (elect_one) {
    asm volatile(
        "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::"
        "complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%2, "
        "%3, %4, %5}], [%6], %7, %8;\n" ::"r"(urb0),
        "l"(reinterpret_cast<uint64_t>(p_desc)), "r"(urb2), "r"(urb3),
        "r"(urb4), "r"(urb5), "r"(urb1 & MMA_PEER_BIT_MASK), "h"(mcast_mask),
        "l"(mem_desc)
        : "memory");
  }
}
#endif

inline __device__ void
utmastg_atomicAdd_4d_tiled(const void *p_desc, uint32_t smem_offset,
                           int32_t urb0, // m
                           int32_t urb1, // n
                           int32_t urb2, // h
                           int32_t urb3, // b
                           const uint32_t elect_one,
                           uint64_t mem_desc = MEM_DESC_DEFAULT) {
  if (elect_one) {
    asm volatile(
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.bulk_group [%0, "
        "{%1, %2, %3, %4}], [%5];\n" ::"l"(reinterpret_cast<uint64_t>(p_desc)),
        "r"(urb0), "r"(urb1), "r"(urb2), "r"(urb3), "r"(smem_offset)
        : "memory");
  }
}

inline __device__ void utmastg_4d_tiled(const void *p_desc,
                                        uint32_t smem_offset,
                                        int32_t urb0, // m
                                        int32_t urb1, // n
                                        int32_t urb2, // h
                                        int32_t urb3, // b
                                        uint64_t mem_desc = MEM_DESC_DEFAULT) {
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%1, %2, %3, "
      "%4}], [%5];\n" ::"l"(reinterpret_cast<uint64_t>(p_desc)),
      "r"(urb0), "r"(urb1), "r"(urb2), "r"(urb3), "r"(smem_offset)
      : "memory");
}

inline __device__ void utmastg_5d_tiled(const void *p_desc,
                                        uint32_t smem_offset,
                                        int32_t urb0, // m
                                        int32_t urb1, // n
                                        int32_t urb2, // h
                                        int32_t urb3, // b
                                        int32_t urb4, // lean_tile_id
                                        uint64_t mem_desc = MEM_DESC_DEFAULT) {
  asm volatile(
      "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%1, %2, %3, "
      "%4, %5}], [%6];\n" ::"l"(reinterpret_cast<uint64_t>(p_desc)),
      "r"(urb0), "r"(urb1), "r"(urb2), "r"(urb3), "r"(urb4), "r"(smem_offset)
      : "memory");
}

// mma_loop_op 2 includes

// global_load_shared_store_op 4 includes

// shared_load_op 5 includes

// shared_load_op 6 includes

// global_load_shared_store_op 7 includes

// mma_op 8 includes

// global_load_op 10 includes

// pointwise_calc_op 9 includes

// softmax_op 12 includes

// global_store_op 13 includes

// global_store_op 14 includes

// shared_load_op 15 includes

// global_load_shared_store_op 16 includes

// mma_op 17 includes

// global_store_op 20 includes

// output_loop_op 3 includes

// shared_store_op 18 includes

// shared_load_op 19 includes
} // namespace fort
)KERNEL"
    R"KERNEL(

// receive_op 0 types
using namespace fort;

static constexpr int THREADS_PER_WARP_0 = 32;
static constexpr int WARPS_PER_GROUP_0 = FORT_MIN(4, 512 / THREADS_PER_WARP_0);
static constexpr int THREADS_PER_WARP_GROUP_0 =
    THREADS_PER_WARP_0 * WARPS_PER_GROUP_0;
static constexpr int BITS_PER_REGISTER_0 = 32; // NOTE: sizeof(uint)
static constexpr int BYTES_PER_REGISTER_0 = BITS_PER_REGISTER_0 / 8;
static constexpr int THREADS_PER_GROUP_0 =
    WARPS_PER_GROUP_0 * THREADS_PER_WARP_0;

// mma_pipeline_op 1 types
static constexpr int BYTES_PER_BANK_1 = 16;
static constexpr int BITS_PER_ELEMENT_1 = 16;
static constexpr int ELEMS_PER_BANK_1 =
    BYTES_PER_BANK_1 * 8 / BITS_PER_ELEMENT_1;
static constexpr int BYTES_PER_ELEMENT_1 = BITS_PER_ELEMENT_1 / 8;
static constexpr int BYTES_PER_FP16_1 = 2;
static constexpr int ELEMS_PER_VECTOR_1 =
    BYTES_PER_REGISTER_0 / BYTES_PER_ELEMENT_1;
static constexpr int BITS_PER_ACC_1 = 32;
static constexpr int BYTES_PER_ACC_1 = BITS_PER_ACC_1 / 8;
static constexpr int TILE_M_1 = 128;
static constexpr int TILE_N_1 = 128;
static constexpr int TILE_K_1 = 64;
static constexpr int TILE_O_1 = FORT_MIN(TILE_K_1, 128);
static constexpr int TILE_V_1 = TILE_K_1;
static constexpr int TILE_DV_1 = FORT_MIN(TILE_K_1, 128);
static constexpr int WARP_TILE_M_1 = 16;
static constexpr int WARP_TILE_N_1 = 16;
static constexpr int WARP_TILE_K_1 = 32 / BYTES_PER_ELEMENT_1;
static constexpr int WARPS_M_1 = 64 / 16;
static constexpr int WARPS_N_1 = 16 / 16;
static constexpr int WARPS_K_1 = 16 / 16;
static constexpr int STAGES_1 = 1;
static constexpr int GROUPS_M_1 = 1;
static constexpr int GMMA_TILE_M_1 = WARPS_M_1 * WARP_TILE_M_1;
static constexpr int GMMA_TILE_K_1 = WARPS_K_1 * WARP_TILE_K_1;
static constexpr int WARP_TILES_M_1 = TILE_M_1 / WARP_TILE_M_1;
static constexpr int WARP_TILES_N_1 = TILE_N_1 / WARP_TILE_N_1;
static constexpr int WARP_TILES_K_1 = TILE_K_1 / WARP_TILE_K_1;
static constexpr int WARP_REGS_1 = TILE_N_1;
static constexpr int REGS_M_1 = WARP_TILES_M_1 / WARPS_M_1;
static constexpr int REGS_N_1 = WARP_TILES_N_1 / WARPS_N_1;
static constexpr int REGS_K_1 = WARP_TILES_K_1 / WARPS_K_1;
static constexpr int REGS_K_FOR_DP_1 = TILE_V_1 / 16;
static constexpr int REGS_dQ_1 = TILE_O_1 / 16;
static constexpr int REGS_O_1 = REGS_K_1 * 1;

static constexpr int CGA_M_1 = 1;
static constexpr int CGA_N_1 = 1;
static constexpr int CTA_MMA_1 = 1;
static constexpr int M_TILES_PER_OUTPUT_TILE_1 = 4;
static constexpr int N_TILES_PER_OUTPUT_TILE_1 = 1;

static constexpr int CTA_TILE_M_1 = TILE_M_1;
static constexpr int CTA_TILE_N_1 = 64;
static constexpr int CTA_TILE_K_1 = 128;

static constexpr float ln2 = 0.6931471805599453094f;
static constexpr float inv_ln2 = 1.4426950408889634074f;

static constexpr uint32_t SchedulerPipelineStageCount = 1;
static constexpr uint32_t AccumulatorPipelineStageCount = 1;
static constexpr uint32_t SmemPipelineStageCount = 1;
static constexpr uint32_t num_columns_per_tmem = 512;
static constexpr uint32_t columns_per_allocation_slice = 32;
static constexpr int stages_q = 2;
static constexpr int stages_kv = 2 * CTA_MMA_1;
static constexpr int Tiles_Q = 2;

static constexpr int SoftmaxWarpGroups = 2;
static constexpr int CorrectionWarpGroups = 1;
static constexpr int SoftmaxWarps = SoftmaxWarpGroups * 4;
static constexpr int CorrectionWarps = 4;

static constexpr int MmaWarpRegs = 40;        // warp 12
static constexpr int TmaldgWarpRegs = 40;     // warp 13
static constexpr int TmastgWarpRegs = 40;     // warp 14
static constexpr int SchedulerWarpRegs = 40;  // warp 15
static constexpr int CorrectionWarpRegs = 88; // warps 8-11
static constexpr int SoftmaxWarpRegs = 192;   // warps 0-7

static constexpr int threads_per_warp = 32;
static constexpr int warps_per_group = 4;
static constexpr int threads_per_group = threads_per_warp * warps_per_group;
static constexpr int threads_per_cta =
    threads_per_warp * (warps_per_group + SoftmaxWarps + CorrectionWarps);

static constexpr int BUFFERS_Q = stages_q;
static constexpr int BUFFERS_K = stages_kv;
static constexpr int BUFFERS_V = stages_kv;
static constexpr int BUFFERS_S = stages_kv;

using ElementQ = fort::bfloat16_t;
using ElementK = fort::bfloat16_t;
using ElementV = fort::bfloat16_t;
using ElementS = fort::bfloat16_t;
using ElementO = fort::bfloat16_t;
using ElementAccumulator = float;

static constexpr int BYTES_PER_ELEMENT = 2;
static constexpr int BYTES_PER_ACC = 4;

static constexpr int BMM1_TILE_M = TILE_M_1;
static constexpr int BMM1_TILE_N = 128;
static constexpr int BMM1_TILE_K = 64;

static constexpr int BMM2_TILE_M = TILE_M_1;
static constexpr int BMM2_TILE_N = 64;
static constexpr int BMM2_TILE_K = 128;

static constexpr uint32_t qBufferElems = BMM1_TILE_M * BMM1_TILE_K;
static constexpr uint32_t kBufferElems = BMM1_TILE_N * BMM1_TILE_K / CTA_MMA_1;
static constexpr uint32_t vBufferElems = BMM2_TILE_N * BMM2_TILE_K / CTA_MMA_1;
static constexpr uint32_t oBufferElems =
    FORT_MAX(8, BMM2_TILE_M) * (BMM2_TILE_N + 0);
static constexpr uint32_t sBufferElems = FORT_MAX(8, BMM1_TILE_M) * BMM1_TILE_N;

static constexpr uint32_t qTmaTransactionBytes =
    qBufferElems * sizeof(ElementQ) * CTA_MMA_1;
static constexpr uint32_t kTmaTransactionBytes =
    kBufferElems * sizeof(ElementK) * CTA_MMA_1;
static constexpr uint32_t vTmaTransactionBytes =
    vBufferElems * sizeof(ElementV) * CTA_MMA_1;

static constexpr int UTCMMA_TILE_M = 128;
static constexpr int UTCMMA_TILE_N = 128;
static constexpr int UTCMMA_TILE_K = 32 / BYTES_PER_ELEMENT;

static constexpr int BMM1_XMMAS_K = BMM1_TILE_K / UTCMMA_TILE_K;
static constexpr int BMM2_XMMAS_K = BMM2_TILE_K / UTCMMA_TILE_K;
static constexpr int BYTES_PER_MMA_K = UTCMMA_TILE_K * BYTES_PER_ELEMENT;

static constexpr int SMEM_BUFFER_SIZE_Q =
    BMM1_TILE_M * BMM1_TILE_K * BYTES_PER_ELEMENT;
static constexpr int SMEM_BUFFER_SIZE_K =
    BMM1_TILE_N * BMM1_TILE_K * BYTES_PER_ELEMENT / CTA_MMA_1;
static constexpr int SMEM_BUFFER_SIZE_V =
    BMM2_TILE_N * BMM2_TILE_K * BYTES_PER_ELEMENT / CTA_MMA_1;
static constexpr int SMEM_BUFFER_SIZE_S =
    FORT_MAX(8, BMM1_TILE_M) * BMM1_TILE_N * BYTES_PER_ELEMENT;

static constexpr int SOFTMAX_BARRIER = 1;
static constexpr int BAND_BIAS_BARRIER = 5;
static constexpr int TMEM_ALLOC_SOFTMAX_BARRIER = 6;
static constexpr int TMEM_ALLOC_CORRECTION_BARRIER = 7;

struct SharedStorage {
  alignas(1024) ElementQ smem_Q[qBufferElems * stages_q];
  alignas(1024) ElementK smem_K[kBufferElems * stages_kv];
  alignas(1024) ElementV smem_V[vBufferElems * stages_kv];
  alignas(1024) ElementO smem_O[oBufferElems];

  alignas(16) uint64_t tma_o_0_empty_mbar[1];

  alignas(16) uint64_t tma_k_empty_mbar[stages_kv];
  alignas(16) uint64_t tma_k_full_mbar[stages_kv];

  alignas(16) uint64_t tma_v_empty_mbar[stages_kv];
  alignas(16) uint64_t tma_v_full_mbar[stages_kv];

  alignas(16) uint64_t tma_q_empty_mbar[stages_q];
  alignas(16) uint64_t tma_q_full_mbar[stages_q];

  alignas(16) uint64_t bmm2_tile0_done_mbar[1];
  alignas(16) uint64_t bmm2_tile0_ready_mbar[2];

  alignas(16) uint64_t stat_tile0_full_mbar[1];
  alignas(16) uint64_t stat_tile0_empty_mbar[1];
  alignas(16) uint64_t final_stat_tile0_empty_mbar[1];
  alignas(16) uint64_t
      tma_o_0_full_mbar[FORT_DIV_UP(TILE_O_1 * sizeof(ElementO), 128)];
  alignas(16) uint64_t
      tma_o_1_full_mbar[FORT_DIV_UP(TILE_O_1 * sizeof(ElementO), 128)];

  alignas(16) uint64_t epilogue_done_mbar[1];
  alignas(16) uint64_t tmaldg_tile_started_mbar[1];

  alignas(16) uint64_t bmm1_tile0_done_mbar[1];
  alignas(16) uint64_t bmm1_tile1_done_mbar[1];

  alignas(16) uint64_t bmm2_tile1_done_mbar[1];
  alignas(16) uint64_t bmm2_tile1_ready_mbar[2];

  alignas(16) uint64_t stat_tile1_full_mbar[1];
  alignas(16) uint64_t stat_tile1_empty_mbar[1];
  alignas(16) uint64_t final_stat_tile1_empty_mbar[1];

  alignas(16) uint64_t empty_mainloop_mbar[1];

  alignas(16) uint64_t scheduler_mbar[SchedulerPipelineStageCount];
  alignas(16) uint64_t read_tile_id_done_mbar[SchedulerPipelineStageCount];
  alignas(16) uint64_t tmem_dealloc_smem_bar[1];

  uint32_t tmem_base_ptrs;

  alignas(16) uint32_t tile_id[8 * SchedulerPipelineStageCount];
};

// global_load_shared_store_op 4 types
static constexpr int BITS_PER_ELEMENT_4 = 16;
static constexpr int BYTES_PER_ELEMENT_4 = BITS_PER_ELEMENT_4 / 8;
static constexpr int BYTES_PER_ACCESS_4 = 16;
static constexpr int TILE_M_4 = 128;
static constexpr int TILE_N_4 = 64;
static constexpr int BYTES_PER_SMEM_4 =
    TILE_M_4 * TILE_N_4 * BYTES_PER_ELEMENT_4;

// shared_load_op 5 types
static constexpr int BITS_PER_ELEMENT_5 = 16;
static constexpr int BYTES_PER_ELEMENT_5 = 2;
static constexpr int BYTES_PER_QUAD_5 = 16;
static constexpr int WARP_TILE_M_5 = 16;
static constexpr int WARP_TILE_N_5 = 16;
static constexpr int TILE_M_5 = 128;
static constexpr int TILE_N_5 = 64;
static constexpr int BYTES_PER_SMEM_5 =
    TILE_M_5 * TILE_N_5 * BYTES_PER_ELEMENT_5;
static constexpr int WARPS_M_5 = 4;
static constexpr int WARPS_N_5 = 1;
static constexpr int WARP_TILES_M_5 = TILE_M_5 / WARP_TILE_M_5;
static constexpr int WARP_TILES_N_5 = TILE_N_5 / WARP_TILE_N_5;
static constexpr int THREADS_PER_WARP_TILE_M_5 = 8;
static constexpr int THREADS_PER_WARP_TILE_N_5 = 4;
static constexpr int REGS_M_5 = WARP_TILES_M_5 / WARPS_M_5;
static constexpr int REGS_N_5 = WARP_TILES_N_5 / WARPS_N_5;
static constexpr int BYTES_PER_LD_5 = 128;
static constexpr int SWIZZLE_SCALE_5 =
    FORT_MIN(BYTES_PER_LD_5 / 16, WARPS_PER_GROUP_0 * 4);

// shared_load_op 6 types
static constexpr int BITS_PER_ELEMENT_6 = 16;
static constexpr int BYTES_PER_ELEMENT_6 = 2;
static constexpr int BYTES_PER_QUAD_6 = 16;
static constexpr int WARP_TILE_M_6 = 16;
static constexpr int WARP_TILE_N_6 = 16;
static constexpr int TILE_M_6 = 128;
static constexpr int TILE_N_6 = 64;
static constexpr int BYTES_PER_SMEM_6 =
    TILE_M_6 * TILE_N_6 * BYTES_PER_ELEMENT_6;
static constexpr int WARPS_M_6 = 1;
static constexpr int WARPS_N_6 = 1;
static constexpr int WARP_TILES_M_6 = TILE_M_6 / WARP_TILE_M_6;
static constexpr int WARP_TILES_N_6 = TILE_N_6 / WARP_TILE_N_6;
static constexpr int THREADS_PER_WARP_TILE_M_6 = 8;
static constexpr int THREADS_PER_WARP_TILE_N_6 = 4;
static constexpr int REGS_M_6 = WARP_TILES_M_6 / WARPS_M_6;
static constexpr int REGS_N_6 = WARP_TILES_N_6 / WARPS_N_6;
static constexpr int BYTES_PER_LD_6 = 128;
static constexpr int SWIZZLE_SCALE_6 =
    FORT_MIN(BYTES_PER_LD_6 / 16, WARPS_PER_GROUP_0 * 4);

// global_load_shared_store_op 7 types
static constexpr int BITS_PER_ELEMENT_7 = 16;
static constexpr int BYTES_PER_ELEMENT_7 = BITS_PER_ELEMENT_7 / 8;
static constexpr int BYTES_PER_ACCESS_7 = 16;
static constexpr int TILE_M_7 = 128;
static constexpr int TILE_N_7 = 64;
static constexpr int BYTES_PER_SMEM_7 =
    TILE_M_7 * TILE_N_7 * BYTES_PER_ELEMENT_7;

// mma_op 8 types
static constexpr int ROWS_PER_CORE_MATRIX_A_8 = 8;
static constexpr int COLS_PER_CORE_MATRIX_A_8 = 8;
static constexpr int ROWS_PER_CORE_MATRIX_B_8 = 8;
static constexpr int COLS_PER_CORE_MATRIX_B_8 = 8;

// global_load_op 10 types
static constexpr int BITS_PER_ELEMENT_10 = 32;
static constexpr int BYTES_PER_ELEMENT_10 = 4;
static constexpr int REGISTERS_PER_VECTOR_10 = 1;
static constexpr int REGISTERS_PER_ACCESS_10 = 1;

// pointwise_calc_op 9 types
static constexpr int WARP_TILE_M_9 = 16;
static constexpr int WARP_TILE_N_9 = 16;
static constexpr int TILE_M_9 = 128;
static constexpr int TILE_N_9 = 128;
static constexpr int WARPS_M_9 = 4;
static constexpr int WARPS_N_9 = 1;
static constexpr int WARP_TILES_M_9 = TILE_M_9 / WARP_TILE_M_9;
static constexpr int WARP_TILES_N_9 = TILE_N_9 / WARP_TILE_N_9;
static constexpr int WARP_REGS_9 = WARP_REGS_1;
static constexpr int REGS_M_9 = WARP_TILES_M_9 / WARPS_M_9;
static constexpr int REGS_N_9 = WARP_TILES_N_9 / WARPS_N_9;

// mha_mask_op 11 types
static constexpr int WARP_TILE_M_11 = 16;
static constexpr int WARP_TILE_N_11 = 16;
static constexpr int TILE_M_11 = 128;
static constexpr int TILE_N_11 = 128;
static constexpr int WARPS_M_11 = 4;
static constexpr int WARPS_N_11 = 1;
static constexpr int WARP_TILES_M_11 = TILE_M_11 / WARP_TILE_M_11;
static constexpr int WARP_TILES_N_11 = TILE_N_11 / WARP_TILE_N_11;
static constexpr int WARP_REGS_11 = 8;
static constexpr int REGS_M_11 = WARP_TILES_M_11 / WARPS_M_11;
static constexpr int REGS_N_11 = WARP_TILES_N_11 / WARPS_N_11;
inline __device__ bool compute_diagonal_band_mask_11(
)KERNEL"
    R"KERNEL(
    const int row, const int col, const int actual_seqlen_kv,
    const int actual_seqlen_q, const int shift_right_bound,
    const int left_bound) {
  constexpr bool is_bottom_right_alignment = false;
  constexpr bool is_shift_right_bound = false;
  constexpr bool is_right_bound = true;
  constexpr bool is_left_bound = false;
  constexpr bool need_oob_check = false;

  const int diag = is_bottom_right_alignment
                       ? row + (actual_seqlen_kv - actual_seqlen_q)
                       : row;
  const int shifted_right_bound =
      is_shift_right_bound ? diag + shift_right_bound : diag;
  const bool right_bound_mask =
      is_right_bound ? col <= shifted_right_bound : true;
  const bool left_bound_mask = is_left_bound ? col + left_bound > diag : true;
  const bool oob_check =
      need_oob_check ? (row < actual_seqlen_q && col < actual_seqlen_kv) : true;
  return right_bound_mask && left_bound_mask && oob_check;
}

// softmax_op 12 types
static constexpr int BYTES_PER_ELEMENT_12 = 2;
static constexpr int WARP_TILE_M_12 = 16;
static constexpr int WARP_TILE_N_12 = 16;
static constexpr int TILE_M_12 = 128;
static constexpr int TILE_N_12 = 128;
static constexpr int WARPS_M_12 = 4;
static constexpr int WARPS_N_12 = 1;
static constexpr int WARP_TILES_M_12 = TILE_M_12 / WARP_TILE_M_12;
static constexpr int WARP_TILES_N_12 = TILE_N_12 / WARP_TILE_N_12;
static constexpr int THREADS_PER_WARP_TILE_M_12 = 8;
static constexpr int THREADS_PER_WARP_TILE_N_12 = 4;
static constexpr int WARP_REGS_12 = 8;
static constexpr int REGS_M_12 = WARP_TILES_M_12 / WARPS_M_12;
static constexpr int REGS_N_12 = WARP_TILES_N_12 / WARPS_N_12;
static constexpr int ROWS_PER_THREAD_12 =
    TILE_M_12 / WARPS_M_12 / THREADS_PER_WARP_TILE_M_12;

// global_store_op 13 types
static constexpr int BYTES_PER_ELEMENT_13 = 4;
static constexpr int WARP_TILE_M_13 = 16;
static constexpr int WARP_TILE_N_13 = 16;
static constexpr int TILE_M_13 = 128;
static constexpr int TILE_N_13 = 128;
static constexpr int WARPS_M_13 = 4;
static constexpr int WARPS_N_13 = 1;
static constexpr int WARP_TILES_M_13 = TILE_M_13 / WARP_TILE_M_13;
static constexpr int WARP_TILES_N_13 = TILE_N_13 / WARP_TILE_N_13;
static constexpr int THREADS_PER_WARP_TILE_M_13 = 8;
static constexpr int THREADS_PER_WARP_TILE_N_13 = 4;
static constexpr int ROWS_PER_THREAD_13 =
    TILE_M_13 / WARPS_M_13 / THREADS_PER_WARP_TILE_M_13;

// global_store_op 14 types
static constexpr int BYTES_PER_ELEMENT_14 = 4;
static constexpr int WARP_TILE_M_14 = 16;
static constexpr int WARP_TILE_N_14 = 16;
static constexpr int TILE_M_14 = 128;
static constexpr int TILE_N_14 = 128;
static constexpr int WARPS_M_14 = 4;
static constexpr int WARPS_N_14 = 1;
static constexpr int WARP_TILES_M_14 = TILE_M_14 / WARP_TILE_M_14;
static constexpr int WARP_TILES_N_14 = TILE_N_14 / WARP_TILE_N_14;
static constexpr int THREADS_PER_WARP_TILE_M_14 = 8;
static constexpr int THREADS_PER_WARP_TILE_N_14 = 4;
static constexpr int ROWS_PER_THREAD_14 =
    TILE_M_14 / WARPS_M_14 / THREADS_PER_WARP_TILE_M_14;

// shared_load_op 15 types
static constexpr int BITS_PER_ELEMENT_15 = 16;
static constexpr int BYTES_PER_ELEMENT_15 = 2;
static constexpr int BYTES_PER_QUAD_15 = 16;
static constexpr int WARP_TILE_M_15 = 16;
static constexpr int WARP_TILE_N_15 = 16;
static constexpr int TILE_M_15 = 128;
static constexpr int TILE_N_15 = 64;
static constexpr int BYTES_PER_SMEM_15 =
    TILE_M_15 * TILE_N_15 * BYTES_PER_ELEMENT_15;
static constexpr int WARPS_M_15 = 1;
static constexpr int WARPS_N_15 = 1;
static constexpr int WARP_TILES_M_15 = TILE_M_15 / WARP_TILE_M_15;
static constexpr int WARP_TILES_N_15 = TILE_N_15 / WARP_TILE_N_15;
static constexpr int THREADS_PER_WARP_TILE_M_15 = 8;
static constexpr int THREADS_PER_WARP_TILE_N_15 = 4;
static constexpr int REGS_M_15 = WARP_TILES_M_15 / WARPS_M_15;
static constexpr int REGS_N_15 = WARP_TILES_N_15 / WARPS_N_15;
static constexpr int BYTES_PER_LD_15 = 128;
static constexpr int SWIZZLE_SCALE_15 =
    FORT_MIN(BYTES_PER_LD_15 / 16, WARPS_PER_GROUP_0 * 4);

// global_load_shared_store_op 16 types
static constexpr int BITS_PER_ELEMENT_16 = 16;
static constexpr int BYTES_PER_ELEMENT_16 = BITS_PER_ELEMENT_16 / 8;
static constexpr int BYTES_PER_ACCESS_16 = 16;
static constexpr int TILE_M_16 = 128;
static constexpr int TILE_N_16 = 64;
static constexpr int BYTES_PER_SMEM_16 =
    TILE_M_16 * TILE_N_16 * BYTES_PER_ELEMENT_16;

// mma_op 17 types
static constexpr int ROWS_PER_CORE_MATRIX_A_17 = 8;
static constexpr int COLS_PER_CORE_MATRIX_A_17 = 8;
static constexpr int ROWS_PER_CORE_MATRIX_B_17 = 8;
static constexpr int COLS_PER_CORE_MATRIX_B_17 = 8;

// global_store_op 20 types

// mma_loop_op 2 types
inline __device__ int2 compute_kv_loop_bounds(
    const int row_coord, const int ROW_TILE_SIZE, const int COL_TILE_SIZE,
    const int actual_seqlen_kv, const int actual_seqlen_q,
    const int shift_right_bound, const int left_bound) {
  constexpr bool is_right_bound = true;
  constexpr bool is_right_bound_bottom_right_alignment = false;
  constexpr bool is_shift_right_bound = false;
  constexpr bool is_left_bound = false;
  constexpr bool is_left_bound_bottom_right_alignment = false;

  const int right_bound_diagonal =
      is_right_bound_bottom_right_alignment
          ? row_coord + (actual_seqlen_kv - actual_seqlen_q)
          : row_coord;
  const int shifted_right_bound_diagonal =
      is_shift_right_bound ? right_bound_diagonal + shift_right_bound
                           : right_bound_diagonal;
  const int left_bound_diagonal =
      is_left_bound_bottom_right_alignment
          ? row_coord + (actual_seqlen_kv - actual_seqlen_q)
          : row_coord;

  const int kv_loop_left_bound =
      is_left_bound
          ? FORT_MAX(0, (left_bound_diagonal - left_bound) / COL_TILE_SIZE)
          : 0;
  const int kv_loop_right_bound =
      is_right_bound
          ? FORT_MIN(FORT_DIV_UP(shifted_right_bound_diagonal + ROW_TILE_SIZE,
                                 COL_TILE_SIZE),
                     FORT_DIV_UP(actual_seqlen_kv, COL_TILE_SIZE))
          : FORT_DIV_UP(actual_seqlen_kv, COL_TILE_SIZE);
  return make_int2(kv_loop_left_bound, kv_loop_right_bound);
}

// output_loop_op 3 types
static constexpr int BYTES_PER_BANK_3 = 16;
static constexpr int ELEMENTS_PER_VECTOR_3 =
    FORT_MIN(64, CTA_TILE_N_1 / N_TILES_PER_OUTPUT_TILE_1);

// shared_store_op 18 types

// shared_load_op 19 types
static constexpr int BITS_PER_ELEMENT_19 = 32 * 1;
static constexpr int BYTES_PER_ELEMENT_19 = BITS_PER_ELEMENT_19 / 8;
static constexpr int BITS_PER_VECTOR_19 =
    BITS_PER_ELEMENT_19 * ELEMENTS_PER_VECTOR_3;
static constexpr int BYTES_PER_VECTOR_19 = BITS_PER_VECTOR_19 / 8;
static constexpr int REGISTERS_PER_VECTOR_19 =
    BITS_PER_VECTOR_19 / BITS_PER_REGISTER_0;

// receive_op 0 code

extern "C" __global__
__launch_bounds__(512, 1) void cudnn_generated_oss_sdpa_sm100_flash_fprop_f16_knob_7_128x128x64_4x1x1_cga1x1x1_kernel0_0(
    const AttentionDescriptor_t attnDesc, const FastDivisor_t num_head_divmod,
    const FastDivisor_t l2_minor_divmod, const FastDivisor_t l2_major_divmod,
    const FastDivisor_t l2_minor_residual_divmod, const int num_hb_quotient,
    const int num_block, const int total_blocks,
    __grid_constant__ const CUtensorMap tma_tensor_3,
    __grid_constant__ const CUtensorMap tma_tensor_2, float tensor_4,
    float tensor_13, void *tensor_7, fort::tensor_descriptor desc_7,
    void *tensor_6, fort::tensor_descriptor desc_6,
    __grid_constant__ const CUtensorMap tma_tensor_1,
    __grid_constant__ const CUtensorMap tma_tensor_5) {
  asm volatile(".pragma \"global knob ForceLateCommoning=1\";\n" ::: "memory");
  asm volatile(".pragma \"global knob HoistLate=3\";\n" ::: "memory");

  extern __shared__ char smem_[];
  SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_);

  // receive_op 0 decls

  // mma_pipeline_op 1 decls
  const int tid = threadIdx.x % threads_per_group;
  const int wid = threadIdx.x / 32;
  const int tiw = threadIdx.x % 32;

  const uint32_t elect_one = elect_one_sync();

  if (wid == 0 && elect_one) {
    smem_bar_init(cast_smem_ptr_to_uint(&shared_storage.tma_o_0_empty_mbar[0]),
                  /* 1 tmastg */ 32);
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.tmem_dealloc_smem_bar[0]),
        /* 4 epilogue */ warps_per_group * 32 * CTA_MMA_1);

#pragma unroll
    for (int q_stage = 0; q_stage < stages_q; q_stage++) {
      smem_bar_init(
          cast_smem_ptr_to_uint(&shared_storage.tma_q_empty_mbar[q_stage]),
          /* 1 mma */ 1);
      smem_bar_init(
          cast_smem_ptr_to_uint(&shared_storage.tma_q_full_mbar[q_stage]),
          /* 1 tma */ 1);
    }
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.bmm2_tile0_ready_mbar[0]),
        /* 4 Softmax */ warps_per_group * 32 * 2 * CTA_MMA_1);
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.bmm2_tile0_done_mbar[0]),
        /* 1 mma */ 1);
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.stat_tile0_full_mbar[0]),
        /* 4 Softmax */ warps_per_group * 32);
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.stat_tile0_empty_mbar[0]),
        /* 4 Softmax */ warps_per_group * 32);
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.final_stat_tile0_empty_mbar[0]),
        /* 4 Correction */ warps_per_group * 32 * CTA_MMA_1);

#pragma unroll
    for (int i = 0; i < TILE_O_1 * sizeof(ElementO); i += 128) {
      smem_bar_init(
          cast_smem_ptr_to_uint(&shared_storage.tma_o_0_full_mbar[i / 128]),
          /* 1 correction */ warps_per_group * 32);
      smem_bar_init(
          cast_smem_ptr_to_uint(&shared_storage.tma_o_1_full_mbar[i / 128]),
          /* 1 correction */ warps_per_group * 32);
    }
    smem_bar_init(cast_smem_ptr_to_uint(&shared_storage.epilogue_done_mbar[0]),
                  /* 1 tmastg */ 32);
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.tmaldg_tile_started_mbar[0]), 32);
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.bmm2_tile1_ready_mbar[0]),
        /* 4 Softmax */ warps_per_group * 32 * 2 * CTA_MMA_1);
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.bmm2_tile0_ready_mbar[1]),
        /* 4 Softmax */ warps_per_group * 32 * 1 * CTA_MMA_1);
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.bmm2_tile1_ready_mbar[1]),
        /* 4 Softmax */ warps_per_group * 32 * 1 * CTA_MMA_1);
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.bmm1_tile0_done_mbar[0]),
        /* 1 mma */ 1);
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.bmm1_tile1_done_mbar[0]),
        /* 1 mma */ 1);
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.bmm2_tile1_done_mbar[0]),
        /* 1 mma */ 1);
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.stat_tile1_full_mbar[0]),
        /* 4 Softmax */ warps_per_group * 32);
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.stat_tile1_empty_mbar[0]),
        /* 4 Softmax */ warps_per_group * 32);
    smem_bar_init(
        cast_smem_ptr_to_uint(&shared_storage.final_stat_tile1_empty_mbar[0]),
        /* 4 Correction */ warps_per_group * 32 * CTA_MMA_1);

    smem_bar_init(cast_smem_ptr_to_uint(&shared_storage.empty_mainloop_mbar[0]),
                  /* 4 Softmax */ warps_per_group * 32 * 1 * CTA_MMA_1);

#pragma unroll
    for (int sched_steps = 0; sched_steps < SchedulerPipelineStageCount;
         sched_steps++) {
      smem_bar_init(
          cast_smem_ptr_to_uint(&shared_storage.scheduler_mbar[sched_steps]),
          /* 1 sched */ 1);
      smem_bar_init(cast_smem_ptr_to_uint(
                        &shared_storage.read_tile_id_done_mbar[sched_steps]),
                    (SoftmaxWarps + CorrectionWarps + 1 + 1) * CGA_M_1 *
                            CGA_N_1 +
                        CGA_M_1 * CGA_N_1 / CTA_MMA_1);
    }

#pragma unroll
    for (int kv_stages = 0; kv_stages < stages_kv; kv_stages++) {
      smem_bar_init(
          cast_smem_ptr_to_uint(&shared_storage.tma_k_empty_mbar[kv_stages]),
          /* 1 mma */ CGA_M_1 / CTA_MMA_1 + CGA_N_1 - 1);
      smem_bar_init(
          cast_smem_ptr_to_uint(&shared_storage.tma_k_full_mbar[kv_stages]),
          /* 1 tma/32 ldgsts */ 1);
      smem_bar_init(
          cast_smem_ptr_to_uint(&shared_storage.tma_v_full_mbar[kv_stages]),
          /* 1 tma/32 ldgsts */ 1);
      smem_bar_init(
          cast_smem_ptr_to_uint(&shared_storage.tma_v_empty_mbar[kv_stages]),
          /* 1 mma */ CGA_M_1 / CTA_MMA_1 + CGA_N_1 - 1);
    }
  }
  __syncthreads();

)KERNEL"
    R"KERNEL(
  // mma_pipeline_op 1 code

  // mma_loop_op 2 code
  uint32_t tile_id_and_coord[8] = {0};

  int actual_seqlen_kv_1 = attnDesc.s_kv;
  int actual_seqlen_q_1 = attnDesc.s_q;
  int q_row_coord = 0;
  int kv_loop_right_bound, kv_loop_left_bound;

  const int left_bound = 0;
  const int shift_right_bound = 0;

  uint32_t tile_idx = blockIdx.x;
  uint32_t bidhb, l2_mod;
  uint32_t block = 0, bidhb_residual = 0;

  fastDivMod(l2_major_divmod, tile_idx, bidhb, l2_mod);
  if (bidhb < num_hb_quotient) {
    {
      fastDivMod(l2_minor_divmod, l2_mod, block, bidhb_residual);
    }
  } else {
    {
      fastDivMod(l2_minor_residual_divmod, l2_mod, block, bidhb_residual);
    }
  }

  int bidhb_actual = bidhb * l2_minor_divmod.val + bidhb_residual;
  uint32_t batch_idx, head_idx;
  fastDivMod(num_head_divmod, bidhb_actual, batch_idx, head_idx);
  block = num_block - 1 - block;
  tile_id_and_coord[0] = head_idx;
  tile_id_and_coord[1] = batch_idx;
  tile_id_and_coord[2] = block;
  tile_id_and_coord[3] = 1;

  int head_coord_q_offset_1 = 0;
  uint32_t head_coord_1, head_coord_k_1, head_coord_v_1;
  uint32_t blocked_row_coord = tile_id_and_coord[2];
  uint32_t head_coord_from_grid_1 = tile_id_and_coord[0];
  uint32_t batch_coord_1 = tile_id_and_coord[1];
  uint32_t lean_tile_id = tile_id_and_coord[0];

  uint32_t is_valid_tile = true;

  head_coord_1 = head_coord_from_grid_1;
  head_coord_k_1 = head_coord_1 / attnDesc.q_heads_per_k;
  head_coord_v_1 = head_coord_1 / attnDesc.q_heads_per_v;

  q_row_coord = (blocked_row_coord * Tiles_Q * BMM1_TILE_M);
  int2 kv_loop_bounds = compute_kv_loop_bounds(
      q_row_coord, BMM1_TILE_M * Tiles_Q * CGA_M_1, TILE_N_1,
      actual_seqlen_kv_1, actual_seqlen_q_1, shift_right_bound, left_bound);
  kv_loop_left_bound = kv_loop_bounds.x;
  kv_loop_right_bound = kv_loop_bounds.y;
  if (actual_seqlen_q_1 == 0 || q_row_coord >= actual_seqlen_q_1) {
    kv_loop_right_bound = kv_loop_left_bound;
  }

  PipelineState<SchedulerPipelineStageCount> sched_state(0, 0); // index, state

  // thread_specialize_op 21 code
  if (wid >= 0 && wid < SoftmaxWarps) {
    // Softmax
    reg_alloc<SoftmaxWarpRegs>();

    const uint32_t softmax_gid = wid / 4;
    const uint32_t softmax_gid_s_offset = softmax_gid * BMM1_TILE_N;

    uint32_t base_tmem_addr;
    named_barrier_wait(TMEM_ALLOC_SOFTMAX_BARRIER,
                       threads_per_warp * (SoftmaxWarps + 1));
    lds_32(&base_tmem_addr,
           cast_smem_ptr_to_uint(&shared_storage.tmem_base_ptrs));
    uint32_t tmem_fp32_S = base_tmem_addr + 0 + softmax_gid_s_offset;
    uint32_t tmem_fp16_S =
        base_tmem_addr + BMM1_TILE_N / 2 + softmax_gid_s_offset;
    uint32_t tmem_Stats = base_tmem_addr + 0 + softmax_gid_s_offset;

    r32 reg_8_0[BMM1_TILE_N];
    r32 fp32_stats[2];

    // Memory barrier states
    uint32_t bmm_mbar_state = 0;
    uint32_t stat_mbar_state = 0;
    uint32_t epilogue_state = 0;

// Persistent loop over tiles
#pragma unroll 1
    while (is_valid_tile) {
      int p_row_1 = q_row_coord + softmax_gid * TILE_M_1 + tid;

      // global_load_op 10 decls
      r32 reg_10_0[1];
      reg_10_0[0] = reinterpret_cast<const r32 &>(tensor_4);
      const int oob_M_10 = 1;
      const int oob_N_10 = 1;

      // pointwise_calc_op 9 decls

      // mha_mask_op 11 decls

      // softmax_op 12 decls
      r32 reg_12_0[32];
      float2 total_sum = {0.0f, 0.0f};
      float2 local_sum = {0.0f, 0.0f};
      float total_max = NEG_INFINITY;
      float actual_total_max = NEG_INFINITY;
      static constexpr float inv_ln2_12 = 1.4426950408889634074f;
      float total_max_scale_12 = inv_ln2_12;
      const float bmm_scale = reinterpret_cast<const float &>(tensor_4);
      total_max_scale_12 *= bmm_scale;

      uint64_t &bmm1_mbar = softmax_gid == 0
                                ? shared_storage.bmm1_tile0_done_mbar[0]
                                : shared_storage.bmm1_tile1_done_mbar[0];
      uint64_t &stat_empty_mbar = softmax_gid == 0
                                      ? shared_storage.stat_tile0_empty_mbar[0]
                                      : shared_storage.stat_tile1_empty_mbar[0];
      uint64_t *bmm2_mbar = softmax_gid == 0
                                ? &(shared_storage.bmm2_tile0_ready_mbar[0])
                                : &(shared_storage.bmm2_tile1_ready_mbar[0]);
      uint64_t *stat_full_mbar =
          softmax_gid == 0 ? &(shared_storage.stat_tile0_full_mbar[0])
                           : &(shared_storage.stat_tile1_full_mbar[0]);

      epilogue_state ^= 1;
      wait_barrier(cast_smem_ptr_to_uint(&shared_storage.epilogue_done_mbar[0]),
                   epilogue_state);
      stat_mbar_state ^= 1;
      wait_barrier(cast_smem_ptr_to_uint(&stat_empty_mbar), stat_mbar_state);

#pragma unroll 1
      for (int kv_loop = kv_loop_left_bound;
           kv_loop <
           kv_loop_right_bound -
               FORT_DIV_UP(BMM1_TILE_M * CGA_M_1 * Tiles_Q, BMM1_TILE_N);
           kv_loop += 1) {
        int p_col_1 = kv_loop * TILE_N_1;
        {

          wait_barrier(cast_smem_ptr_to_uint(&bmm1_mbar), bmm_mbar_state);

          r32 stat_reg[BMM1_TILE_N / 64];
#pragma unroll
          for (int ldtm_step = 0; ldtm_step < BMM1_TILE_N / 32; ldtm_step++) {
            ldtm_32dp32bit_x32(&reg_8_0[32 * ldtm_step],
                               tmem_fp32_S + ldtm_step * 32);
          }
          constexpr bool use_ldtm_stat_max = false;

          // global_load_op 10 code

          // softmax_op 12 code
          float current_max =
              row_max_reduction_128_elems(reg_8_0) * total_max_scale_12;

          // Update total max
          float total_max_tmp = fmaxf(current_max, total_max);
          current_max = total_max;
          total_max = total_max_tmp;

          float2 bmm_scale_x_ln2_x2 =
              make_float2(total_max_scale_12, total_max_scale_12);
          float total_max_scaled =
              (total_max == NEG_INFINITY) ? 0.0f : total_max;
          float2 minus_scaled_max_x2 =
              make_float2(-total_max_scaled, -total_max_scaled);

          static constexpr int kMulPipeCount = 14;      // must be multiple of 2
          static constexpr int kSubtractPipeCount = 12; // must be multiple of 2
          static constexpr int kFmaPipeCount = 12;      // must be multiple of 2
          static constexpr int kConvertPipeCount = 8;   // must be multiple of 2
          static constexpr int kReleasePipeCount = 4;   // must be multiple of 2
          static constexpr int kDropoutPipeCount =
              6; // must be multiple of 2 and less than or equal
                 // kConvertPipeCount
          static constexpr int kAddPipeCount =
              4; // must be multiple of 2 and less than kDropoutPipeCount

          // To better overlap EX2 MUFU and emulation, we need to have last
          // kE2eRes of every kE2eFreq elements in a row to use emulation while
          // the rest go to MUFU. Also, we need to avoid using emulation after
          // kE2eLimit elements as it would delay the execution of row sum of
          // another softmax warpgroup because of resource contention
          static constexpr int kE2eFreq = 16;
          static constexpr int kE2eRes = 8;
          static constexpr int kE2eLimit = BMM1_TILE_N / 16 * 9;

#pragma unroll
          for (int i = 0; i < kFmaPipeCount; i += 2) {
            float2 in = make_float2(reinterpret_cast<float &>(reg_8_0[i]),
                                    reinterpret_cast<float &>(reg_8_0[i + 1]));
            float2 out = ffma2(bmm_scale_x_ln2_x2, in, minus_scaled_max_x2);

            reinterpret_cast<float &>(reg_8_0[i + 0]) = out.x;
            reinterpret_cast<float &>(reg_8_0[i + 1]) = out.y;
          }

          // Optimized version of alpha  = __expf(current_max - total_max);
          float exp_input = (total_max == NEG_INFINITY)
                                ? NEG_INFINITY
                                : (current_max - total_max);
          float alpha = exp2f(exp_input);

          // Send stats to correction warp
          reinterpret_cast<float &>(fp32_stats[0]) = alpha;

          // STTM
          sttm_32dp32bit_x1(tmem_Stats, &fp32_stats[0]);
          fence_view_async_tmem_store();
          arrive_barrier(cast_smem_ptr_to_uint(stat_full_mbar));

#pragma unroll
          for (int i = 0; i < BMM1_TILE_N; i += 2) {
            if (i - kConvertPipeCount == 32) {
              sttm_32dp32bit_x16(tmem_fp16_S, &reg_12_0[0]);
            }
            if (i - kConvertPipeCount == 64) {
              sttm_32dp32bit_x16(tmem_fp16_S + 16, &reg_12_0[16]);
              fence_view_async_tmem_store();
              arrive_barrier(cast_smem_ptr_to_uint(bmm2_mbar + 0));
            }

            if (i >= kConvertPipeCount) {
              fp32x2_to_bf16x2(
                  &reg_12_0[((i - kConvertPipeCount) / 2) % 32],
                  reinterpret_cast<r32 *>(&reg_8_0[i - kConvertPipeCount]));
            }

            reinterpret_cast<float &>(reg_8_0[i + 0]) =
                exp2f(reinterpret_cast<float &>(reg_8_0[i + 0]));


            if (i + kFmaPipeCount < BMM1_TILE_N) {
              float2 in = make_float2(
                  reinterpret_cast<float &>(reg_8_0[i + kFmaPipeCount + 0]),
                  reinterpret_cast<float &>(reg_8_0[i + kFmaPipeCount + 1]));
              float2 out = ffma2(bmm_scale_x_ln2_x2, in, minus_scaled_max_x2);

              reinterpret_cast<float &>(reg_8_0[i + kFmaPipeCount + 0]) = out.x;
              reinterpret_cast<float &>(reg_8_0[i + kFmaPipeCount + 1]) = out.y;
            }

            reinterpret_cast<float &>(reg_8_0[i + 1]) =
                exp2f(reinterpret_cast<float &>(reg_8_0[i + 1]));

            if (i == kE2eLimit) {
              if (softmax_gid == 1) {
                named_barrier_arrive(SOFTMAX_BARRIER, 256);
                named_barrier_wait(SOFTMAX_BARRIER + 1, 256);
              }
            }
          }

#pragma unroll
          for (int i = BMM1_TILE_N - kConvertPipeCount; i < BMM1_TILE_N;
               i += 2) {
            fp32x2_to_bf16x2(&reg_12_0[(i / 2) % 32],
                             reinterpret_cast<r32 *>(&reg_8_0[i]));
          }
          sttm_32dp32bit_x32(tmem_fp16_S + BMM1_TILE_N / 64 * 16,
                             &reg_12_0[((BMM1_TILE_N / 64) % 2) * 16]);
          fence_view_async_tmem_store();
          arrive_barrier(
              cast_smem_ptr_to_uint(bmm2_mbar + (BMM1_TILE_N / 64 - 1)));

          stat_mbar_state ^= 1;
          if (!smem_bar_peek(cast_smem_ptr_to_uint(&stat_empty_mbar),
                             stat_mbar_state)) {
            wait_barrier(cast_smem_ptr_to_uint(&stat_empty_mbar),
                         stat_mbar_state);
          }

          float2 alpha2 = make_float2(alpha, alpha);
          float local_row_sum_0 = 0.0f;
          total_sum = fmul2(alpha2, total_sum);

#pragma unroll
          for (int i = 0; i < BMM1_TILE_N; i += 2) {
            float2 in = make_float2(reinterpret_cast<float &>(reg_8_0[i + 0]),
                                    reinterpret_cast<float &>(reg_8_0[i + 1]));
            total_sum = fadd2(total_sum, in);
          }
          if (softmax_gid == 0) {
            named_barrier_arrive(SOFTMAX_BARRIER + 1, 256);
            named_barrier_wait(SOFTMAX_BARRIER, 256);
          }
          named_barrier_wait(SOFTMAX_BARRIER + 2 + softmax_gid, 128);
        }
        bmm_mbar_state ^= 1;
      } // End of mainloop
#pragma unroll 1
      for (int kv_loop = FORT_MAX(
               kv_loop_right_bound -
                   FORT_DIV_UP(BMM1_TILE_M * CGA_M_1 * Tiles_Q, BMM1_TILE_N),
               kv_loop_left_bound);
           kv_loop < kv_loop_right_bound; kv_loop += 1) {
        int p_col_1 = kv_loop * TILE_N_1;
        {

          wait_barrier(cast_smem_ptr_to_uint(&bmm1_mbar), bmm_mbar_state);

          r32 stat_reg[BMM1_TILE_N / 64];
#pragma unroll
          for (int ldtm_step = 0; ldtm_step < BMM1_TILE_N / 32; ldtm_step++) {
            ldtm_32dp32bit_x32(&reg_8_0[32 * ldtm_step],
                               tmem_fp32_S + ldtm_step * 32);
          }
          constexpr bool use_ldtm_stat_max = false;

)KERNEL"
    R"KERNEL(
          // global_load_op 10 code

          // mha_mask_op 11 code

          {
#pragma unroll
            for (int i = 0; i < BMM1_TILE_N; i++) {
              int row = p_row_1;
              int col = p_col_1 + i;
              bool mask = compute_diagonal_band_mask_11(
                  row, col, actual_seqlen_kv_1, actual_seqlen_q_1,
                  shift_right_bound, left_bound);
              if (!mask) {
                reinterpret_cast<float &>(reg_8_0[i]) = tensor_13;
              }
            }
          }

          // softmax_op 12 code
          float current_max =
              row_max_reduction_128_elems(reg_8_0) * total_max_scale_12;

          // Update total max
          float total_max_tmp = fmaxf(current_max, total_max);
          current_max = total_max;
          total_max = total_max_tmp;

          float2 bmm_scale_x_ln2_x2 =
              make_float2(total_max_scale_12, total_max_scale_12);
          float total_max_scaled =
              (total_max == NEG_INFINITY) ? 0.0f : total_max;
          float2 minus_scaled_max_x2 =
              make_float2(-total_max_scaled, -total_max_scaled);

          static constexpr int kMulPipeCount = 14;      // must be multiple of 2
          static constexpr int kSubtractPipeCount = 12; // must be multiple of 2
          static constexpr int kFmaPipeCount = 12;      // must be multiple of 2
          static constexpr int kConvertPipeCount = 8;   // must be multiple of 2
          static constexpr int kReleasePipeCount = 4;   // must be multiple of 2
          static constexpr int kDropoutPipeCount =
              6; // must be multiple of 2 and less than or equal
                 // kConvertPipeCount
          static constexpr int kAddPipeCount =
              4; // must be multiple of 2 and less than kDropoutPipeCount

          // To better overlap EX2 MUFU and emulation, we need to have last
          // kE2eRes of every kE2eFreq elements in a row to use emulation while
          // the rest go to MUFU. Also, we need to avoid using emulation after
          // kE2eLimit elements as it would delay the execution of row sum of
          // another softmax warpgroup because of resource contention
          static constexpr int kE2eFreq = 16;
          static constexpr int kE2eRes = 8;
          static constexpr int kE2eLimit = BMM1_TILE_N / 16 * 9;

#pragma unroll
          for (int i = 0; i < kFmaPipeCount; i += 2) {
            float2 in = make_float2(reinterpret_cast<float &>(reg_8_0[i]),
                                    reinterpret_cast<float &>(reg_8_0[i + 1]));
            float2 out = ffma2(bmm_scale_x_ln2_x2, in, minus_scaled_max_x2);

            reinterpret_cast<float &>(reg_8_0[i + 0]) = out.x;
            reinterpret_cast<float &>(reg_8_0[i + 1]) = out.y;
          }

          // Optimized version of alpha  = __expf(current_max - total_max);
          float exp_input = (total_max == NEG_INFINITY)
                                ? NEG_INFINITY
                                : (current_max - total_max);
          float alpha = exp2f(exp_input);

          // Send stats to correction warp
          reinterpret_cast<float &>(fp32_stats[0]) = alpha;

          // STTM
          sttm_32dp32bit_x1(tmem_Stats, &fp32_stats[0]);
          fence_view_async_tmem_store();
          arrive_barrier(cast_smem_ptr_to_uint(stat_full_mbar));

#pragma unroll
          for (int i = 0; i < BMM1_TILE_N; i += 2) {
            if (i - kConvertPipeCount == 32) {
              sttm_32dp32bit_x16(tmem_fp16_S, &reg_12_0[0]);
            }
            if (i - kConvertPipeCount == 64) {
              sttm_32dp32bit_x16(tmem_fp16_S + 16, &reg_12_0[16]);
              fence_view_async_tmem_store();
              arrive_barrier(cast_smem_ptr_to_uint(bmm2_mbar + 0));
            }

            if (i >= kConvertPipeCount) {
              fp32x2_to_bf16x2(
                  &reg_12_0[((i - kConvertPipeCount) / 2) % 32],
                  reinterpret_cast<r32 *>(&reg_8_0[i - kConvertPipeCount]));
            }

            reinterpret_cast<float &>(reg_8_0[i + 0]) =
                exp2f(reinterpret_cast<float &>(reg_8_0[i + 0]));


            if (i + kFmaPipeCount < BMM1_TILE_N) {
              float2 in = make_float2(
                  reinterpret_cast<float &>(reg_8_0[i + kFmaPipeCount + 0]),
                  reinterpret_cast<float &>(reg_8_0[i + kFmaPipeCount + 1]));
              float2 out = ffma2(bmm_scale_x_ln2_x2, in, minus_scaled_max_x2);

              reinterpret_cast<float &>(reg_8_0[i + kFmaPipeCount + 0]) = out.x;
              reinterpret_cast<float &>(reg_8_0[i + kFmaPipeCount + 1]) = out.y;
            }

            reinterpret_cast<float &>(reg_8_0[i + 1]) =
                exp2f(reinterpret_cast<float &>(reg_8_0[i + 1]));

            if (i == kE2eLimit) {
              if (softmax_gid == 1) {
                named_barrier_arrive(SOFTMAX_BARRIER, 256);
                named_barrier_wait(SOFTMAX_BARRIER + 1, 256);
              }
            }
          }

#pragma unroll
          for (int i = BMM1_TILE_N - kConvertPipeCount; i < BMM1_TILE_N;
               i += 2) {
            fp32x2_to_bf16x2(&reg_12_0[(i / 2) % 32],
                             reinterpret_cast<r32 *>(&reg_8_0[i]));
          }
          sttm_32dp32bit_x32(tmem_fp16_S + BMM1_TILE_N / 64 * 16,
                             &reg_12_0[((BMM1_TILE_N / 64) % 2) * 16]);
          fence_view_async_tmem_store();
          arrive_barrier(
              cast_smem_ptr_to_uint(bmm2_mbar + (BMM1_TILE_N / 64 - 1)));

          stat_mbar_state ^= 1;
          if (!smem_bar_peek(cast_smem_ptr_to_uint(&stat_empty_mbar),
                             stat_mbar_state)) {
            wait_barrier(cast_smem_ptr_to_uint(&stat_empty_mbar),
                         stat_mbar_state);
          }

          float2 alpha2 = make_float2(alpha, alpha);
          float local_row_sum_0 = 0.0f;
          total_sum = fmul2(alpha2, total_sum);

#pragma unroll
          for (int i = 0; i < BMM1_TILE_N; i += 2) {
            float2 in = make_float2(reinterpret_cast<float &>(reg_8_0[i + 0]),
                                    reinterpret_cast<float &>(reg_8_0[i + 1]));
            total_sum = fadd2(total_sum, in);
          }
          if (softmax_gid == 0) {
            named_barrier_arrive(SOFTMAX_BARRIER + 1, 256);
            named_barrier_wait(SOFTMAX_BARRIER, 256);
          }
          named_barrier_wait(SOFTMAX_BARRIER + 2 + softmax_gid, 128);
        }
        bmm_mbar_state ^= 1;
      } // End of mainloop

      float final_total_sum = total_sum.x + total_sum.y;

      reinterpret_cast<float &>(fp32_stats[0]) = total_max;
      reinterpret_cast<float &>(fp32_stats[1]) = final_total_sum;
      // STTM
      sttm_32dp32bit_x2(tmem_Stats, &fp32_stats[0]);
      fence_view_async_tmem_store();
      arrive_barrier(cast_smem_ptr_to_uint(stat_full_mbar));

      // This portion of the code may seem strange, please read the
      // documentation here:
      // Function: work_tile_info_from_workid_response, load_query_response
      if (elect_one) {
        arrive_barrier(cast_smem_ptr_to_uint(
            &(shared_storage.read_tile_id_done_mbar[sched_state.index()])));
      }
      // Get next tile id
      wait_barrier(cast_smem_ptr_to_uint(
                       &shared_storage.scheduler_mbar[sched_state.index()]),
                   sched_state.phase());
      uint32_t tile_id_smem_int_ptr = cast_smem_ptr_to_uint(
          &shared_storage.tile_id[sched_state.index() * 8]);
      lds_128(tile_id_and_coord, tile_id_smem_int_ptr);
      ++sched_state;

      uint32_t tile_idx = tile_id_and_coord[0];
      uint32_t bidhb, l2_mod;
      uint32_t block = 0, bidhb_residual = 0;

      fastDivMod(l2_major_divmod, tile_idx, bidhb, l2_mod);
      if (bidhb < num_hb_quotient) {
        {
          fastDivMod(l2_minor_divmod, l2_mod, block, bidhb_residual);
        }
      } else {
        {
          fastDivMod(l2_minor_residual_divmod, l2_mod, block, bidhb_residual);
        }
      }

      int bidhb_actual = bidhb * l2_minor_divmod.val + bidhb_residual;
      uint32_t batch_idx, head_idx;
      fastDivMod(num_head_divmod, bidhb_actual, batch_idx, head_idx);
      block = num_block - 1 - block;

      lean_tile_id = tile_id_and_coord[0];

      blocked_row_coord = block + 0;
      head_coord_from_grid_1 = (head_idx) + 0;
      batch_coord_1 = batch_idx;
      is_valid_tile = tile_id_and_coord[2] & 1;

      head_coord_1 = head_coord_from_grid_1;
      head_coord_k_1 = head_coord_1 / attnDesc.q_heads_per_k;
      head_coord_v_1 = head_coord_1 / attnDesc.q_heads_per_v;

      q_row_coord = (blocked_row_coord * Tiles_Q * BMM1_TILE_M);
      int2 kv_loop_bounds = compute_kv_loop_bounds(
          q_row_coord, BMM1_TILE_M * Tiles_Q * CGA_M_1, TILE_N_1,
          actual_seqlen_kv_1, actual_seqlen_q_1, shift_right_bound, left_bound);
      kv_loop_left_bound = kv_loop_bounds.x;
      kv_loop_right_bound = kv_loop_bounds.y;
      if (actual_seqlen_q_1 == 0 || q_row_coord >= actual_seqlen_q_1) {
        kv_loop_right_bound = kv_loop_left_bound;
      }
    } // end of persistent loop
  }

  // thread_specialize_op 22 code
  else if (wid >= SoftmaxWarps && wid < SoftmaxWarps + CorrectionWarps) {
    // Correction
    reg_dealloc<CorrectionWarpRegs>();
    uint32_t base_tmem_addr;
    named_barrier_wait(TMEM_ALLOC_CORRECTION_BARRIER,
                       threads_per_warp * (CorrectionWarps + 1));
    lds_32(&base_tmem_addr,
           cast_smem_ptr_to_uint(&shared_storage.tmem_base_ptrs));
    uint32_t tmem_fp32_O_0 = base_tmem_addr + 256;
    uint32_t tmem_fp32_O_1 = base_tmem_addr + 384;
    uint32_t tmem_Stats_0 = base_tmem_addr + 0;
    uint32_t tmem_Stats_1 = base_tmem_addr + 0 + BMM1_TILE_N;

    static constexpr int local_correction_block_size = 16;
    static constexpr int local_epilogue_block_size =
        128 / sizeof(ElementO); // hack as only K=128/64 supported. Both can do
                                // 128B TMASTG.

)KERNEL"
    R"KERNEL(
    float amax_s_1 = 0.f;
    float amax_o_1 = 0.f;

    // Registers
    uint32_t fp32_stats[2];
    uint32_t fp32_O[local_correction_block_size];

    // Memory barrier states
    uint32_t epilogue_state = 0;
    uint32_t bmm_mbar_state = 0;
    uint32_t stat_mbar_state = 0;

// Persistent loop over tiles
#pragma unroll 1
    while (is_valid_tile) {
      if (elect_one) {
        arrive_barrier(cast_smem_ptr_to_uint(
            &(shared_storage.read_tile_id_done_mbar[sched_state.index()])));
      }

      if (kv_loop_left_bound < kv_loop_right_bound) {
        // No correction for iter 0
        arrive_barrier(
            cast_smem_ptr_to_uint(&(shared_storage.bmm2_tile0_ready_mbar[0])));
        arrive_barrier(
            cast_smem_ptr_to_uint(&(shared_storage.bmm2_tile1_ready_mbar[0])));

        {
          wait_barrier(
              cast_smem_ptr_to_uint(&shared_storage.stat_tile0_full_mbar[0]),
              stat_mbar_state);
          arrive_barrier(cast_smem_ptr_to_uint(
              &(shared_storage.stat_tile0_empty_mbar[0])));
        }
        {
          wait_barrier(
              cast_smem_ptr_to_uint(&shared_storage.stat_tile1_full_mbar[0]),
              stat_mbar_state);
          arrive_barrier(cast_smem_ptr_to_uint(
              &(shared_storage.stat_tile1_empty_mbar[0])));
        }
        stat_mbar_state ^= 1;
      } else {
        arrive_barrier(
            cast_smem_ptr_to_uint(&(shared_storage.empty_mainloop_mbar[0])));
      }

#pragma unroll 1
      for (int kv_loop = kv_loop_left_bound + 1; kv_loop < kv_loop_right_bound;
           kv_loop += 1) {
#pragma unroll
        for (int sub_tile_id = 0; sub_tile_id < 2; ++sub_tile_id) {
          uint32_t tmem_Stats = sub_tile_id == 0 ? tmem_Stats_0 : tmem_Stats_1;
          uint32_t tmem_fp32_O =
              sub_tile_id == 0 ? tmem_fp32_O_0 : tmem_fp32_O_1;

          uint64_t &stat_full_mbar =
              sub_tile_id == 0 ? shared_storage.stat_tile0_full_mbar[0]
                               : shared_storage.stat_tile1_full_mbar[0];
          uint64_t &bmm_done_mbar =
              sub_tile_id == 0 ? shared_storage.bmm2_tile0_done_mbar[0]
                               : shared_storage.bmm2_tile1_done_mbar[0];
          uint64_t *stat_empty_mbar =
              sub_tile_id == 0 ? &(shared_storage.stat_tile0_empty_mbar[0])
                               : &(shared_storage.stat_tile1_empty_mbar[0]);
          uint64_t *bmm_ready_mbar =
              sub_tile_id == 0 ? &(shared_storage.bmm2_tile0_ready_mbar[0])
                               : &(shared_storage.bmm2_tile1_ready_mbar[0]);

          wait_barrier(cast_smem_ptr_to_uint(&stat_full_mbar), stat_mbar_state);
          ldtm_32dp32bit_x1(&fp32_stats[0], tmem_Stats);

          float alpha = reinterpret_cast<float &>(fp32_stats[0]);
          fence_view_async_tmem_load();
          arrive_barrier(cast_smem_ptr_to_uint(stat_empty_mbar));

          bool alpha_is_one = (alpha == 1.0f);
          bool all_alpha_one = __all_sync(0xFFFFFFFF, alpha_is_one);

          float2 alpha2 = make_float2(alpha, alpha);

          wait_barrier(cast_smem_ptr_to_uint(&bmm_done_mbar), bmm_mbar_state);

          if (!all_alpha_one) {
#pragma unroll
            for (int block = 0; block < TILE_O_1 / local_correction_block_size;
                 block++) {
// LDTM
#pragma unroll
              for (int ldtm_step = 0;
                   ldtm_step < local_correction_block_size / 8; ldtm_step++) {
                ldtm_32dp32bit_x8(&fp32_O[8 * ldtm_step],
                                  tmem_fp32_O +
                                      block * local_correction_block_size +
                                      ldtm_step * 8);
              }

#pragma unroll
              for (int i = 0; i < local_correction_block_size; i += 2) {
                float2 in =
                    make_float2(reinterpret_cast<float &>(fp32_O[i + 0]),
                                reinterpret_cast<float &>(fp32_O[i + 1]));
                float2 out = fmul2(alpha2, in);
                reinterpret_cast<float &>(fp32_O[i + 0]) = out.x;
                reinterpret_cast<float &>(fp32_O[i + 1]) = out.y;
              }

#pragma unroll
              for (int sttm_step = 0;
                   sttm_step < local_correction_block_size / 8; sttm_step++) {
                sttm_32dp32bit_x8(tmem_fp32_O +
                                      block * local_correction_block_size +
                                      sttm_step * 8,
                                  &fp32_O[8 * sttm_step]);
              }
            }
          }
          fence_view_async_tmem_store();
          arrive_barrier(cast_smem_ptr_to_uint(bmm_ready_mbar));
        }
        stat_mbar_state ^= 1;
        bmm_mbar_state ^= 1;
      } // End of mainloop

      // Epilogue
      {
        epilogue_state ^= 1;

        static constexpr int stg_step = 16 / sizeof(ElementO);
        const int local_row = (tid % 8) / 1;

        // global_store_op 13 decls
        int oob_M_13 = desc_7.dims[2];
        int row_13 = 0;
        char *ptr_13 =
            reinterpret_cast<char *>(tensor_7) +
            batch_coord_1 * desc_7.strides[0] * BYTES_PER_ELEMENT_13 +
            head_coord_1 * desc_7.strides[1] * BYTES_PER_ELEMENT_13 +
            row_13 * desc_7.strides[2] * BYTES_PER_ELEMENT_13;

        // global_store_op 14 decls
        int oob_M_14 = desc_6.dims[2];
        int row_14 = 0;
        char *ptr_14 =
            reinterpret_cast<char *>(tensor_6) +
            batch_coord_1 * desc_6.strides[0] * BYTES_PER_ELEMENT_14 +
            head_coord_1 * desc_6.strides[1] * BYTES_PER_ELEMENT_14 +
            row_14 * desc_6.strides[2] * BYTES_PER_ELEMENT_14;

#pragma unroll
        for (int sub_tile_id = 0; sub_tile_id < 2; ++sub_tile_id) {
          const int row_coord = q_row_coord + sub_tile_id * TILE_M_1 + tid;
          ElementO *smem_O = (sub_tile_id == 0) ? shared_storage.smem_O
                                                : shared_storage.smem_V;

          uint32_t fp32_reg_O[local_epilogue_block_size];
          uint32_t reg_O[local_epilogue_block_size / (4 / sizeof(ElementO))];

          uint32_t tmem_Stats = sub_tile_id == 0 ? tmem_Stats_0 : tmem_Stats_1;
          uint32_t tmem_fp32_O =
              sub_tile_id == 0 ? tmem_fp32_O_0 : tmem_fp32_O_1;

          uint64_t &stat_full_mbar =
              sub_tile_id == 0 ? shared_storage.stat_tile0_full_mbar[0]
                               : shared_storage.stat_tile1_full_mbar[0];
          uint64_t &bmm_done_mbar =
              sub_tile_id == 0 ? shared_storage.bmm2_tile0_done_mbar[0]
                               : shared_storage.bmm2_tile1_done_mbar[0];
          uint64_t &stat_empty_mbar =
              sub_tile_id == 0 ? shared_storage.stat_tile0_empty_mbar[0]
                               : shared_storage.stat_tile1_empty_mbar[0];
          uint64_t &final_stat_empty_mbar =
              sub_tile_id == 0 ? shared_storage.final_stat_tile0_empty_mbar[0]
                               : shared_storage.final_stat_tile1_empty_mbar[0];

          wait_barrier(cast_smem_ptr_to_uint(&stat_full_mbar), stat_mbar_state);
          ldtm_32dp32bit_x2(&fp32_stats[0], tmem_Stats);

          float total_max = reinterpret_cast<float &>(fp32_stats[0]) * ln2;
          float total_sum = reinterpret_cast<float &>(fp32_stats[1]);
          fence_view_async_tmem_load();
          arrive_barrier(cast_smem_ptr_to_uint(&stat_empty_mbar));
          arrive_barrier(cast_smem_ptr_to_uint(&final_stat_empty_mbar));

          float softmax_stats =
              total_sum == 0.f ? NEG_INFINITY : total_max + __logf(total_sum);

          // global_store_op 13 code
          if (row_coord < actual_seqlen_q_1) {
            stg_32(ptr_13 +
                       row_coord * desc_7.strides[2] * BYTES_PER_ELEMENT_13,
                   &(reinterpret_cast<uint32_t &>(total_max)));
          }

          // global_store_op 14 code
          if (row_coord < actual_seqlen_q_1) {
            stg_32(ptr_14 +
                       row_coord * desc_6.strides[2] * BYTES_PER_ELEMENT_14,
                   &(reinterpret_cast<uint32_t &>(total_sum)));
          }

          float threshold_beta = (total_sum == 0.f) ? 0.f : 1.f / total_sum;
          float beta = threshold_beta;

          float2 beta2 = make_float2(threshold_beta, threshold_beta);

          wait_barrier(cast_smem_ptr_to_uint(&bmm_done_mbar), bmm_mbar_state);

          if (sub_tile_id == 0) {
            wait_barrier(
                cast_smem_ptr_to_uint(&shared_storage.tma_o_0_empty_mbar[0]),
                epilogue_state);
          }
#pragma unroll
          for (int block = 0; block < TILE_O_1 / local_epilogue_block_size;
               block++) {
            {
#pragma unroll
              for (int ldtm_step = 0;
                   ldtm_step < local_epilogue_block_size / 32; ldtm_step++) {
                ldtm_32dp32bit_x32(&fp32_reg_O[32 * ldtm_step],
                                   tmem_fp32_O +
                                       block * local_epilogue_block_size +
                                       ldtm_step * 32);
              }
            }

#pragma unroll
            for (int i = 0; i < local_epilogue_block_size; i += 2) {
              float2 in =
                  make_float2(reinterpret_cast<float &>(fp32_reg_O[i + 0]),
                              reinterpret_cast<float &>(fp32_reg_O[i + 1]));
              float2 out = fmul2(beta2, in);
              reinterpret_cast<float &>(fp32_reg_O[i + 0]) = out.x;
              reinterpret_cast<float &>(fp32_reg_O[i + 1]) = out.y;
            }

// general_data_type_cast code
#pragma unroll
            for (int i = 0; i < local_epilogue_block_size; i += 2) {
              fp32x2_to_bf16x2(&reg_O[i / 2], &fp32_reg_O[i]);
            }

#pragma unroll
            for (int i = 0; i < local_epilogue_block_size / stg_step; i++) {
              char *smem_loc = get_smem_loc_epilogue_swizzle_128b(
                  reinterpret_cast<char *>(smem_O), block, tid, local_row, i,
                  sizeof(ElementO), local_epilogue_block_size, TILE_M_1);
)KERNEL"
    R"KERNEL(
              sts_128(cast_smem_ptr_to_uint(smem_loc),
                      reinterpret_cast<r32 *>(&reg_O[i * 4]));
            }

            uint64_t *tma_o_full_mbar =
                sub_tile_id == 0 ? &(shared_storage.tma_o_0_full_mbar[block])
                                 : &(shared_storage.tma_o_1_full_mbar[block]);
            fence_view_async_shared();
            arrive_barrier(cast_smem_ptr_to_uint(tma_o_full_mbar));
          }
        }
        stat_mbar_state ^= 1;
        bmm_mbar_state ^= 1;
      } // Epilogue end

      // Get next tile id
      wait_barrier(cast_smem_ptr_to_uint(
                       &shared_storage.scheduler_mbar[sched_state.index()]),
                   sched_state.phase());
      uint32_t tile_id_smem_int_ptr = cast_smem_ptr_to_uint(
          &shared_storage.tile_id[sched_state.index() * 8]);
      lds_128(tile_id_and_coord, tile_id_smem_int_ptr);
      ++sched_state;

      uint32_t tile_idx = tile_id_and_coord[0];
      uint32_t bidhb, l2_mod;
      uint32_t block = 0, bidhb_residual = 0;

      fastDivMod(l2_major_divmod, tile_idx, bidhb, l2_mod);
      if (bidhb < num_hb_quotient) {
        {
          fastDivMod(l2_minor_divmod, l2_mod, block, bidhb_residual);
        }
      } else {
        {
          fastDivMod(l2_minor_residual_divmod, l2_mod, block, bidhb_residual);
        }
      }

      int bidhb_actual = bidhb * l2_minor_divmod.val + bidhb_residual;
      uint32_t batch_idx, head_idx;
      fastDivMod(num_head_divmod, bidhb_actual, batch_idx, head_idx);
      block = num_block - 1 - block;

      lean_tile_id = tile_id_and_coord[0];

      blocked_row_coord = block + 0;
      head_coord_from_grid_1 = (head_idx) + 0;
      batch_coord_1 = batch_idx;
      is_valid_tile = tile_id_and_coord[2] & 1;

      head_coord_1 = head_coord_from_grid_1;
      head_coord_k_1 = head_coord_1 / attnDesc.q_heads_per_k;
      head_coord_v_1 = head_coord_1 / attnDesc.q_heads_per_v;

      q_row_coord = (blocked_row_coord * Tiles_Q * BMM1_TILE_M);
      int2 kv_loop_bounds = compute_kv_loop_bounds(
          q_row_coord, BMM1_TILE_M * Tiles_Q * CGA_M_1, TILE_N_1,
          actual_seqlen_kv_1, actual_seqlen_q_1, shift_right_bound, left_bound);
      kv_loop_left_bound = kv_loop_bounds.x;
      kv_loop_right_bound = kv_loop_bounds.y;
      if (actual_seqlen_q_1 == 0 || q_row_coord >= actual_seqlen_q_1) {
        kv_loop_right_bound = kv_loop_left_bound;
      }
    } // end of persistent loop
    arrive_barrier(
        cast_smem_ptr_to_uint(&shared_storage.tmem_dealloc_smem_bar[0]));
  }

  // thread_specialize_op 23 code
  else if (wid == SoftmaxWarps + CorrectionWarps) {
    // MMA
    reg_dealloc<MmaWarpRegs>();
    tmem_allocate_1sm(num_columns_per_tmem,
                      cast_smem_ptr_to_uint(&shared_storage.tmem_base_ptrs));
    tmem_release_allocation_lock_1sm();
    __syncwarp();
    named_barrier_arrive(TMEM_ALLOC_SOFTMAX_BARRIER,
                         threads_per_warp * (SoftmaxWarps + 1));
    named_barrier_arrive(TMEM_ALLOC_CORRECTION_BARRIER,
                         threads_per_warp * (CorrectionWarps + 1));
    uint32_t base_tmem_addr;
    lds_32(&base_tmem_addr,
           cast_smem_ptr_to_uint(&shared_storage.tmem_base_ptrs));

    uint32_t tmem_S_acc_0 = base_tmem_addr + 0;
    uint32_t tmem_S_acc_1 = base_tmem_addr + BMM1_TILE_N;

    uint32_t tmem_S_bmm2_0 = base_tmem_addr + BMM1_TILE_N / 2;
    uint32_t tmem_S_bmm2_1 = base_tmem_addr + 3 * BMM1_TILE_N / 2;

    uint32_t tmem_O_0 = base_tmem_addr + 256;
    uint32_t tmem_O_1 = base_tmem_addr + 384;

    // Memory barrier states
    uint32_t bmm_mbar_state = 0;
    uint32_t final_stat_mbar_state = 1;
    uint32_t empty_mainloop_mbar_state = 0;

    PipelineState<stages_kv> kv_mbar_state(0, 0); // index, state
    PipelineState<stages_q> q_mbar_state(0, 0);   // index, state

    // mma_op 8 decls
    constexpr uint64_t utcmma_instruction_desc_8 =
        build_utcmma_instruction_desc<0, // max shift for WS
                                      TILE_M_1 * CTA_MMA_1, TILE_N_1,
                                      0, // Transpose for B
                                      0, // Transpose for A
                                      0, // Negate for B
                                      0, // Negate for A
                                      1, // Data type B
                                      1, // Data type A
                                      0, // sparse meta data format
                                      1, // Acc type fp32
                                      0, // saturate disable
                                      0, // sparse disable
                                      0  // sparse meta data id2 ???
                                      >();
    // Swizzle mode, base smem offset, desc type, bytes per leading dim, tile k
    Smem_utcmma_descriptor utcmma_smem_desc_Q_0(2, 0, 1, 16,
                                                ROWS_PER_CORE_MATRIX_A_8 * 128);
    Smem_utcmma_descriptor utcmma_smem_desc_K(2, 0, 1, 16,
                                              ROWS_PER_CORE_MATRIX_B_8 * 128);

    utcmma_smem_desc_Q_0.set_smem<SMEM_BUFFER_SIZE_Q, BUFFERS_Q>(
        cast_smem_ptr_to_uint(shared_storage.smem_Q));
    utcmma_smem_desc_K.set_smem<SMEM_BUFFER_SIZE_K, BUFFERS_K>(
        cast_smem_ptr_to_uint(shared_storage.smem_K));

    // mma_op 17 decls
    constexpr uint64_t utcmma_instruction_desc_17 =
        build_utcmma_instruction_desc<0, // max shift for WS
                                      TILE_M_1 * CTA_MMA_1, TILE_O_1,
                                      1, // Transpose for B
                                      0, // Transpose for A
                                      0, // Negate for B
                                      0, // Negate for A
                                      1, // Data type B
                                      1, // Data type A
                                      0, // sparse meta data format
                                      1, // Acc type fp32
                                      0, // saturate disable
                                      0, // sparse disable
                                      0  // sparse meta data id2 ???
                                      >();
    // Swizzle mode, base smem offset, desc type, bytes per leading dim, tile k
    Smem_utcmma_descriptor utcmma_smem_desc_V(
        2, 0, 1, TILE_O_1 / ROWS_PER_CORE_MATRIX_B_17 <= 8 ? 0 : TILE_N_1 * 128,
        COLS_PER_CORE_MATRIX_B_17 * 128);

    utcmma_smem_desc_V.set_smem<SMEM_BUFFER_SIZE_V, BUFFERS_V>(
        cast_smem_ptr_to_uint(shared_storage.smem_V));

// Persistent loop over tiles
#pragma unroll 1
    while (is_valid_tile) {
      if (elect_one) {
        arrive_barrier(cast_smem_ptr_to_uint(
            &(shared_storage.read_tile_id_done_mbar[sched_state.index()])));
      }
      //---------------------------------------------------------//
      //------------------- MMA Pipeline Begin ------------------//
      //---------------------------------------------------------//

      if (kv_loop_left_bound < kv_loop_right_bound) {
        wait_barrier(cast_smem_ptr_to_uint(
                         &shared_storage.final_stat_tile0_empty_mbar[0]),
                     final_stat_mbar_state);

        const int kv_stage = kv_mbar_state.index();
        const int kv_state = kv_mbar_state.phase();
        wait_barrier(
            cast_smem_ptr_to_uint(&shared_storage.tma_k_full_mbar[kv_stage]),
            kv_state);

        const int q_stage = q_mbar_state.index();
        const int q_phase = q_mbar_state.phase();
        wait_barrier(
            cast_smem_ptr_to_uint(&shared_storage.tma_q_full_mbar[q_stage]),
            q_phase);
        ++q_mbar_state;

        // Q * K matmul for sub-tile 0
        {
          Smem_utcmma_descriptor &utcmma_smem_desc_Q = utcmma_smem_desc_Q_0;
          uint32_t tmem_acc = tmem_S_acc_0;

          // mma_op 8 code
#pragma unroll
          for (int k = 0; k < BMM1_XMMAS_K; ++k) {
            if (elect_one)
              utcmma_asmem_bsmem_h_cta1(utcmma_smem_desc_Q.desc,
                                        utcmma_smem_desc_K.desc, tmem_acc,
                                        utcmma_instruction_desc_8, (k > 0));
            if ((k % 4) == ((BMM1_XMMAS_K - 1) % 4)) {
              utcmma_smem_desc_Q.add_smem_offset<-BYTES_PER_MMA_K *(
                  (BMM1_XMMAS_K - 1) % 4)>();
              utcmma_smem_desc_Q.add_smem_offset<128 * BMM1_TILE_M>();
            } else {
              utcmma_smem_desc_Q.add_smem_offset<BYTES_PER_MMA_K>();
            }
            if ((k % 4) == ((BMM1_XMMAS_K - 1) % 4)) {
              utcmma_smem_desc_K.add_smem_offset<-BYTES_PER_MMA_K *(
                  (BMM1_XMMAS_K - 1) % 4)>();
              utcmma_smem_desc_K
                  .add_smem_offset<128 * BMM1_TILE_N / CTA_MMA_1>();
            } else {
              utcmma_smem_desc_K.add_smem_offset<BYTES_PER_MMA_K>();
            }
          }
          utcmma_smem_desc_Q.add_smem_offset<-128 * BMM1_TILE_M *
                                             FORT_MAX(BMM1_XMMAS_K / 4, 1)>();
          utcmma_smem_desc_K.add_smem_offset<-128 * BMM1_TILE_N / CTA_MMA_1 *
                                             FORT_MAX(BMM1_XMMAS_K / 4, 1)>();
          // change the smem buffer for Q
          utcmma_smem_desc_Q
              .increment_smem_buffer<SMEM_BUFFER_SIZE_Q, BUFFERS_Q>();
        }

        if (elect_one) {
          umma_arrive(
              cast_smem_ptr_to_uint(&shared_storage.bmm1_tile0_done_mbar[0]));
        } // elect_one
      } // if (kv_loop_left_bound < kv_loop_right_bound)
      if (kv_loop_left_bound < kv_loop_right_bound) {
        wait_barrier(cast_smem_ptr_to_uint(
                         &shared_storage.final_stat_tile1_empty_mbar[0]),
                     final_stat_mbar_state);

        const int kv_stage = kv_mbar_state.index();
        const int kv_state = kv_mbar_state.phase();
        const int q_stage = q_mbar_state.index();
        const int q_phase = q_mbar_state.phase();
        wait_barrier(
            cast_smem_ptr_to_uint(&shared_storage.tma_q_full_mbar[q_stage]),
            q_phase);
        ++q_mbar_state;

        // Q * K matmul for sub-tile 1
        {
          Smem_utcmma_descriptor &utcmma_smem_desc_Q = utcmma_smem_desc_Q_0;
          uint32_t tmem_acc = tmem_S_acc_1;

          // mma_op 8 code
#pragma unroll
          for (int k = 0; k < BMM1_XMMAS_K; ++k) {
            if (elect_one)
              utcmma_asmem_bsmem_h_cta1(utcmma_smem_desc_Q.desc,
)KERNEL"
    R"KERNEL(
                                        utcmma_smem_desc_K.desc, tmem_acc,
                                        utcmma_instruction_desc_8, (k > 0));
            if ((k % 4) == ((BMM1_XMMAS_K - 1) % 4)) {
              utcmma_smem_desc_Q.add_smem_offset<-BYTES_PER_MMA_K *(
                  (BMM1_XMMAS_K - 1) % 4)>();
              utcmma_smem_desc_Q.add_smem_offset<128 * BMM1_TILE_M>();
            } else {
              utcmma_smem_desc_Q.add_smem_offset<BYTES_PER_MMA_K>();
            }
            if ((k % 4) == ((BMM1_XMMAS_K - 1) % 4)) {
              utcmma_smem_desc_K.add_smem_offset<-BYTES_PER_MMA_K *(
                  (BMM1_XMMAS_K - 1) % 4)>();
              utcmma_smem_desc_K
                  .add_smem_offset<128 * BMM1_TILE_N / CTA_MMA_1>();
            } else {
              utcmma_smem_desc_K.add_smem_offset<BYTES_PER_MMA_K>();
            }
          }
          utcmma_smem_desc_Q.add_smem_offset<-128 * BMM1_TILE_M *
                                             FORT_MAX(BMM1_XMMAS_K / 4, 1)>();
          utcmma_smem_desc_K.add_smem_offset<-128 * BMM1_TILE_N / CTA_MMA_1 *
                                             FORT_MAX(BMM1_XMMAS_K / 4, 1)>();
          // change the smem buffer for K
          utcmma_smem_desc_K
              .increment_smem_buffer<SMEM_BUFFER_SIZE_K, BUFFERS_K>();
          // change the smem buffer for Q
          utcmma_smem_desc_Q
              .increment_smem_buffer<SMEM_BUFFER_SIZE_Q, BUFFERS_Q>();
          const int kv_stage = kv_mbar_state.index();
          if (elect_one)
            umma_arrive(cast_smem_ptr_to_uint(
                &shared_storage.tma_k_empty_mbar[kv_stage]));
        }

        if (elect_one) {
          umma_arrive(
              cast_smem_ptr_to_uint(&shared_storage.bmm1_tile1_done_mbar[0]));
        } // elect_one
      }

// Mainloop
#pragma unroll 1
      for (int kv_loop = kv_loop_left_bound + 1; kv_loop < kv_loop_right_bound;
           kv_loop += 1) {
        const int kv_stage = kv_mbar_state.index();
        const uint32_t kv_state = kv_mbar_state.phase();

        ++kv_mbar_state;

        wait_barrier(
            cast_smem_ptr_to_uint(&shared_storage.tma_v_full_mbar[kv_stage]),
            kv_state);

        // S * V matmul for sub-tile 0
        {
          uint32_t bmm2_ready_mbar =
              cast_smem_ptr_to_uint(&shared_storage.bmm2_tile0_ready_mbar[0]);
          uint32_t tmem_O = tmem_O_0;
          uint32_t tmem_S = tmem_S_bmm2_0;
          uint32_t scaleC = kv_loop > kv_loop_left_bound + 1;

          // mma_op 17 code
#pragma unroll
          for (int k = 0; k < BMM2_XMMAS_K; ++k) {
            if (k % (BMM2_XMMAS_K / (BMM1_TILE_N / 64)) == 0) {
              wait_barrier(bmm2_ready_mbar +
                               (k / (BMM2_XMMAS_K / (BMM1_TILE_N / 64))) * 8,
                           bmm_mbar_state);
            }
            if (elect_one)
              utcmma_atmem_bsmem_h_cta1(
                  tmem_S + k * (UTCMMA_TILE_K / (4 / BYTES_PER_ELEMENT)),
                  utcmma_smem_desc_V.desc, tmem_O, utcmma_instruction_desc_17,
                  scaleC || k > 0);
            if (k == BMM2_XMMAS_K - 1) {
              utcmma_smem_desc_V
                  .add_smem_offset<-UTCMMA_TILE_K * 128 * (BMM2_XMMAS_K - 1)>();
            } else {
              utcmma_smem_desc_V.add_smem_offset<UTCMMA_TILE_K * 128>();
            }
          }
          if (elect_one)
            umma_arrive(
                cast_smem_ptr_to_uint(&shared_storage.bmm2_tile0_done_mbar[0]));
        }

        wait_barrier(
            cast_smem_ptr_to_uint(
                &shared_storage.tma_k_full_mbar[kv_mbar_state.index()]),
            kv_mbar_state.phase());

        // Q * K matmul for sub-tile 0
        {
          Smem_utcmma_descriptor &utcmma_smem_desc_Q = utcmma_smem_desc_Q_0;
          uint32_t tmem_acc = tmem_S_acc_0;

          // mma_op 8 code
#pragma unroll
          for (int k = 0; k < BMM1_XMMAS_K; ++k) {
            if (elect_one)
              utcmma_asmem_bsmem_h_cta1(utcmma_smem_desc_Q.desc,
                                        utcmma_smem_desc_K.desc, tmem_acc,
                                        utcmma_instruction_desc_8, (k > 0));
            if ((k % 4) == ((BMM1_XMMAS_K - 1) % 4)) {
              utcmma_smem_desc_Q.add_smem_offset<-BYTES_PER_MMA_K *(
                  (BMM1_XMMAS_K - 1) % 4)>();
              utcmma_smem_desc_Q.add_smem_offset<128 * BMM1_TILE_M>();
            } else {
              utcmma_smem_desc_Q.add_smem_offset<BYTES_PER_MMA_K>();
            }
            if ((k % 4) == ((BMM1_XMMAS_K - 1) % 4)) {
              utcmma_smem_desc_K.add_smem_offset<-BYTES_PER_MMA_K *(
                  (BMM1_XMMAS_K - 1) % 4)>();
              utcmma_smem_desc_K
                  .add_smem_offset<128 * BMM1_TILE_N / CTA_MMA_1>();
            } else {
              utcmma_smem_desc_K.add_smem_offset<BYTES_PER_MMA_K>();
            }
          }
          utcmma_smem_desc_Q.add_smem_offset<-128 * BMM1_TILE_M *
                                             FORT_MAX(BMM1_XMMAS_K / 4, 1)>();
          utcmma_smem_desc_K.add_smem_offset<-128 * BMM1_TILE_N / CTA_MMA_1 *
                                             FORT_MAX(BMM1_XMMAS_K / 4, 1)>();
          // change the smem buffer for Q
          utcmma_smem_desc_Q
              .increment_smem_buffer<SMEM_BUFFER_SIZE_Q, BUFFERS_Q>();
        }

        if (elect_one) {
          umma_arrive(
              cast_smem_ptr_to_uint(&shared_storage.bmm1_tile0_done_mbar[0]));
        } // elect_one

        // S * V matmul for sub-tile 1
        {
          uint32_t bmm2_ready_mbar =
              cast_smem_ptr_to_uint(&shared_storage.bmm2_tile1_ready_mbar[0]);
          uint32_t tmem_O = tmem_O_1;
          uint32_t tmem_S = tmem_S_bmm2_1;
          uint32_t scaleC = kv_loop > kv_loop_left_bound + 1;

          // mma_op 17 code
#pragma unroll
          for (int k = 0; k < BMM2_XMMAS_K; ++k) {
            if (k % (BMM2_XMMAS_K / (BMM1_TILE_N / 64)) == 0) {
              wait_barrier(bmm2_ready_mbar +
                               (k / (BMM2_XMMAS_K / (BMM1_TILE_N / 64))) * 8,
                           bmm_mbar_state);
            }
            if (elect_one)
              utcmma_atmem_bsmem_h_cta1(
                  tmem_S + k * (UTCMMA_TILE_K / (4 / BYTES_PER_ELEMENT)),
                  utcmma_smem_desc_V.desc, tmem_O, utcmma_instruction_desc_17,
                  scaleC || k > 0);
            if (k == BMM2_XMMAS_K - 1) {
              utcmma_smem_desc_V
                  .add_smem_offset<-UTCMMA_TILE_K * 128 * (BMM2_XMMAS_K - 1)>();
            } else {
              utcmma_smem_desc_V.add_smem_offset<UTCMMA_TILE_K * 128>();
            }
          }
          if (elect_one)
            umma_arrive(
                cast_smem_ptr_to_uint(&shared_storage.bmm2_tile1_done_mbar[0]));
          if (elect_one)
            umma_arrive(cast_smem_ptr_to_uint(
                &shared_storage.tma_v_empty_mbar[kv_stage]));
          // Change the buffer for V
          utcmma_smem_desc_V
              .increment_smem_buffer<SMEM_BUFFER_SIZE_V, BUFFERS_V>();
        }
        bmm_mbar_state ^= 1;

        // Q * K matmul for sub-tile 1
        {
          Smem_utcmma_descriptor &utcmma_smem_desc_Q = utcmma_smem_desc_Q_0;
          uint32_t tmem_acc = tmem_S_acc_1;

          // mma_op 8 code
#pragma unroll
          for (int k = 0; k < BMM1_XMMAS_K; ++k) {
            if (elect_one)
              utcmma_asmem_bsmem_h_cta1(utcmma_smem_desc_Q.desc,
                                        utcmma_smem_desc_K.desc, tmem_acc,
                                        utcmma_instruction_desc_8, (k > 0));
            if ((k % 4) == ((BMM1_XMMAS_K - 1) % 4)) {
              utcmma_smem_desc_Q.add_smem_offset<-BYTES_PER_MMA_K *(
                  (BMM1_XMMAS_K - 1) % 4)>();
              utcmma_smem_desc_Q.add_smem_offset<128 * BMM1_TILE_M>();
            } else {
              utcmma_smem_desc_Q.add_smem_offset<BYTES_PER_MMA_K>();
            }
            if ((k % 4) == ((BMM1_XMMAS_K - 1) % 4)) {
              utcmma_smem_desc_K.add_smem_offset<-BYTES_PER_MMA_K *(
                  (BMM1_XMMAS_K - 1) % 4)>();
              utcmma_smem_desc_K
                  .add_smem_offset<128 * BMM1_TILE_N / CTA_MMA_1>();
            } else {
              utcmma_smem_desc_K.add_smem_offset<BYTES_PER_MMA_K>();
            }
          }
          utcmma_smem_desc_Q.add_smem_offset<-128 * BMM1_TILE_M *
                                             FORT_MAX(BMM1_XMMAS_K / 4, 1)>();
          utcmma_smem_desc_K.add_smem_offset<-128 * BMM1_TILE_N / CTA_MMA_1 *
                                             FORT_MAX(BMM1_XMMAS_K / 4, 1)>();
          // change the smem buffer for K
          utcmma_smem_desc_K
              .increment_smem_buffer<SMEM_BUFFER_SIZE_K, BUFFERS_K>();
          // change the smem buffer for Q
          utcmma_smem_desc_Q
              .increment_smem_buffer<SMEM_BUFFER_SIZE_Q, BUFFERS_Q>();
          const int kv_stage = kv_mbar_state.index();
          if (elect_one)
            umma_arrive(cast_smem_ptr_to_uint(
                &shared_storage.tma_k_empty_mbar[kv_stage]));
        }

        if (elect_one) {
          umma_arrive(
              cast_smem_ptr_to_uint(&shared_storage.bmm1_tile1_done_mbar[0]));
        } // elect_one
      }
      if (kv_loop_left_bound < kv_loop_right_bound) {
        if (elect_one) {
          umma_arrive(
              cast_smem_ptr_to_uint(&shared_storage.tma_q_empty_mbar[1]));
          umma_arrive(
              cast_smem_ptr_to_uint(&shared_storage.tma_q_empty_mbar[0]));
        }
        const int kv_stage = kv_mbar_state.index();
        const uint32_t kv_state = kv_mbar_state.phase();
        ++kv_mbar_state;

        wait_barrier(
            cast_smem_ptr_to_uint(&shared_storage.tma_v_full_mbar[kv_stage]),
            kv_state);

        // S * V matmul for sub-tile 0
        {
          uint32_t bmm2_ready_mbar =
              cast_smem_ptr_to_uint(&shared_storage.bmm2_tile0_ready_mbar[0]);
          uint32_t tmem_O = tmem_O_0;
          uint32_t tmem_S = tmem_S_bmm2_0;
          uint32_t scaleC = kv_loop_right_bound > kv_loop_left_bound + 1;

          // mma_op 17 code
#pragma unroll
          for (int k = 0; k < BMM2_XMMAS_K; ++k) {
            if (k % (BMM2_XMMAS_K / (BMM1_TILE_N / 64)) == 0) {
              wait_barrier(bmm2_ready_mbar +
)KERNEL"
    R"KERNEL(
                               (k / (BMM2_XMMAS_K / (BMM1_TILE_N / 64))) * 8,
                           bmm_mbar_state);
            }
            if (elect_one)
              utcmma_atmem_bsmem_h_cta1(
                  tmem_S + k * (UTCMMA_TILE_K / (4 / BYTES_PER_ELEMENT)),
                  utcmma_smem_desc_V.desc, tmem_O, utcmma_instruction_desc_17,
                  scaleC || k > 0);
            if (k == BMM2_XMMAS_K - 1) {
              utcmma_smem_desc_V
                  .add_smem_offset<-UTCMMA_TILE_K * 128 * (BMM2_XMMAS_K - 1)>();
            } else {
              utcmma_smem_desc_V.add_smem_offset<UTCMMA_TILE_K * 128>();
            }
          }
          if (elect_one)
            umma_arrive(
                cast_smem_ptr_to_uint(&shared_storage.bmm2_tile0_done_mbar[0]));
        }
        // S * V matmul for sub-tile 1
        {
          uint32_t bmm2_ready_mbar =
              cast_smem_ptr_to_uint(&shared_storage.bmm2_tile1_ready_mbar[0]);
          uint32_t tmem_O = tmem_O_1;
          uint32_t tmem_S = tmem_S_bmm2_1;
          uint32_t scaleC = kv_loop_right_bound > kv_loop_left_bound + 1;

          // mma_op 17 code
#pragma unroll
          for (int k = 0; k < BMM2_XMMAS_K; ++k) {
            if (k % (BMM2_XMMAS_K / (BMM1_TILE_N / 64)) == 0) {
              wait_barrier(bmm2_ready_mbar +
                               (k / (BMM2_XMMAS_K / (BMM1_TILE_N / 64))) * 8,
                           bmm_mbar_state);
            }
            if (elect_one)
              utcmma_atmem_bsmem_h_cta1(
                  tmem_S + k * (UTCMMA_TILE_K / (4 / BYTES_PER_ELEMENT)),
                  utcmma_smem_desc_V.desc, tmem_O, utcmma_instruction_desc_17,
                  scaleC || k > 0);
            if (k == BMM2_XMMAS_K - 1) {
              utcmma_smem_desc_V
                  .add_smem_offset<-UTCMMA_TILE_K * 128 * (BMM2_XMMAS_K - 1)>();
            } else {
              utcmma_smem_desc_V.add_smem_offset<UTCMMA_TILE_K * 128>();
            }
          }
          if (elect_one)
            umma_arrive(
                cast_smem_ptr_to_uint(&shared_storage.bmm2_tile1_done_mbar[0]));
          if (elect_one)
            umma_arrive(cast_smem_ptr_to_uint(
                &shared_storage.tma_v_empty_mbar[kv_stage]));
          // Change the buffer for V
          utcmma_smem_desc_V
              .increment_smem_buffer<SMEM_BUFFER_SIZE_V, BUFFERS_V>();
        }
        bmm_mbar_state ^= 1;
      } else {
        wait_barrier(
            cast_smem_ptr_to_uint(&shared_storage.empty_mainloop_mbar[0]),
            empty_mainloop_mbar_state);
        empty_mainloop_mbar_state ^= 1;
        if (elect_one) {
          umma_arrive(
              cast_smem_ptr_to_uint(&shared_storage.bmm2_tile0_done_mbar[0]));
          umma_arrive(
              cast_smem_ptr_to_uint(&shared_storage.bmm2_tile1_done_mbar[0]));
        }
      }
      final_stat_mbar_state ^= 1;
      __syncwarp();

      // Get next tile id
      wait_barrier(cast_smem_ptr_to_uint(
                       &shared_storage.scheduler_mbar[sched_state.index()]),
                   sched_state.phase());
      uint32_t tile_id_smem_int_ptr = cast_smem_ptr_to_uint(
          &shared_storage.tile_id[sched_state.index() * 8]);
      lds_128(tile_id_and_coord, tile_id_smem_int_ptr);
      ++sched_state;

      uint32_t tile_idx = tile_id_and_coord[0];
      uint32_t bidhb, l2_mod;
      uint32_t block = 0, bidhb_residual = 0;

      fastDivMod(l2_major_divmod, tile_idx, bidhb, l2_mod);
      if (bidhb < num_hb_quotient) {
        {
          fastDivMod(l2_minor_divmod, l2_mod, block, bidhb_residual);
        }
      } else {
        {
          fastDivMod(l2_minor_residual_divmod, l2_mod, block, bidhb_residual);
        }
      }

      int bidhb_actual = bidhb * l2_minor_divmod.val + bidhb_residual;
      uint32_t batch_idx, head_idx;
      fastDivMod(num_head_divmod, bidhb_actual, batch_idx, head_idx);
      block = num_block - 1 - block;

      lean_tile_id = tile_id_and_coord[0];

      blocked_row_coord = block + 0;
      head_coord_from_grid_1 = (head_idx) + 0;
      batch_coord_1 = batch_idx;
      is_valid_tile = tile_id_and_coord[2] & 1;

      head_coord_1 = head_coord_from_grid_1;
      head_coord_k_1 = head_coord_1 / attnDesc.q_heads_per_k;
      head_coord_v_1 = head_coord_1 / attnDesc.q_heads_per_v;

      q_row_coord = (blocked_row_coord * Tiles_Q * BMM1_TILE_M);
      int2 kv_loop_bounds = compute_kv_loop_bounds(
          q_row_coord, BMM1_TILE_M * Tiles_Q * CGA_M_1, TILE_N_1,
          actual_seqlen_kv_1, actual_seqlen_q_1, shift_right_bound, left_bound);
      kv_loop_left_bound = kv_loop_bounds.x;
      kv_loop_right_bound = kv_loop_bounds.y;
      if (actual_seqlen_q_1 == 0 || q_row_coord >= actual_seqlen_q_1) {
        kv_loop_right_bound = kv_loop_left_bound;
      }
    }
    wait_barrier(
        cast_smem_ptr_to_uint(&shared_storage.tmem_dealloc_smem_bar[0]), 0);
    tmem_free_1sm(num_columns_per_tmem, shared_storage.tmem_base_ptrs);
  }

  // thread_specialize_op 24 code
  else if (wid == SoftmaxWarps + CorrectionWarps + 1) {
    // TMALDG Q, K, V
    reg_dealloc<TmaldgWarpRegs>();

    // Memory barrier states
    uint32_t epilogue_state = 1;
    PipelineState<stages_q> q_mbar_state(0, 1);   // index, state
    PipelineState<stages_kv> kv_mbar_state(0, 1); // index, state
    PipelineState<2> sts_page_buffer_state(0, 0); // index, state
    PipelineState<2> lds_page_buffer_state(0, 0); // index, state

// Persistent loop over tiles
#pragma unroll 1
    while (is_valid_tile) {
      if (elect_one) {
        arrive_barrier(cast_smem_ptr_to_uint(
            &(shared_storage.read_tile_id_done_mbar[sched_state.index()])));
      }

      // global_load_shared_store_op 4 decls

      // global_load_shared_store_op 7 decls

      // global_load_shared_store_op 16 decls
      if (kv_loop_left_bound < kv_loop_right_bound) {
        int kv_loop = kv_loop_left_bound;

        const int q_stage = q_mbar_state.index();
        const int q_phase = q_mbar_state.phase();
        wait_barrier(
            cast_smem_ptr_to_uint(&shared_storage.tma_q_empty_mbar[q_stage]),
            q_phase);
        ++q_mbar_state;

        // Load Q sub-tile 0
        if (true) {
          smem_bar_set_transaction_count(
              cast_smem_ptr_to_uint(&shared_storage.tma_q_full_mbar[q_stage]),
              qTmaTransactionBytes, elect_one);
        }
        {
          uint32_t local_smem_q = cast_smem_ptr_to_uint(shared_storage.smem_Q +
                                                        q_stage * qBufferElems);
          uint32_t local_smem_bar_tma_q =
              cast_smem_ptr_to_uint(&shared_storage.tma_q_full_mbar[q_stage]);
          const uint32_t row_coord = q_row_coord;

          // global_load_shared_store_op 4 code
#pragma unroll
          for (int i = 0; i < TILE_O_1 * BYTES_PER_ELEMENT_1; i += 128) {
            utmaldg_4d_tiled(&tma_tensor_3, local_smem_q + i * TILE_M_1,
                             local_smem_bar_tma_q, i / BYTES_PER_ELEMENT_1,
                             row_coord, head_coord_1 + 0, batch_coord_1,
                             elect_one);
          }
        }

        // Load K
        const int k_stage = kv_mbar_state.index();
        const int k_phase = kv_mbar_state.phase();
        wait_barrier(
            cast_smem_ptr_to_uint(&shared_storage.tma_k_empty_mbar[k_stage]),
            k_phase);

        if (true) {
          smem_bar_set_transaction_count(
              cast_smem_ptr_to_uint(&shared_storage.tma_k_full_mbar[k_stage]),
              kTmaTransactionBytes, elect_one);
        }
        {
          const int lds_page_buffer_state_idx = lds_page_buffer_state.index();
          uint32_t local_smem_k = cast_smem_ptr_to_uint(shared_storage.smem_K +
                                                        k_stage * kBufferElems);
          uint32_t local_smem_bar_tma_k =
              cast_smem_ptr_to_uint(&shared_storage.tma_k_full_mbar[k_stage]);
          const uint32_t p_col_coord = kv_loop * TILE_N_1;

          // global_load_shared_store_op 7 code
#pragma unroll
          for (int i = 0; i < TILE_O_1 * BYTES_PER_ELEMENT_1; i += 128) {
            utmaldg_4d_tiled(&tma_tensor_2, local_smem_k + i * TILE_N_1,
                             local_smem_bar_tma_k, i / BYTES_PER_ELEMENT_1,
                             p_col_coord, head_coord_k_1, batch_coord_1,
                             elect_one);
          }
        }
      } // if (kv_loop_left_bound < kv_loop_right_bound)

      if (kv_loop_left_bound < kv_loop_right_bound) {
        const int q_stage = q_mbar_state.index();
        const int q_phase = q_mbar_state.phase();
        wait_barrier(
            cast_smem_ptr_to_uint(&shared_storage.tma_q_empty_mbar[q_stage]),
            q_phase);
        ++q_mbar_state;

        // Load Q sub-tile 1
        if (true) {
          smem_bar_set_transaction_count(
              cast_smem_ptr_to_uint(&shared_storage.tma_q_full_mbar[q_stage]),
              qTmaTransactionBytes, elect_one);
        }
        {
          uint32_t local_smem_q = cast_smem_ptr_to_uint(shared_storage.smem_Q +
                                                        q_stage * qBufferElems);
          uint32_t local_smem_bar_tma_q =
              cast_smem_ptr_to_uint(&shared_storage.tma_q_full_mbar[q_stage]);
          const uint32_t row_coord = q_row_coord + TILE_M_1;

          // global_load_shared_store_op 4 code
#pragma unroll
          for (int i = 0; i < TILE_O_1 * BYTES_PER_ELEMENT_1; i += 128) {
            utmaldg_4d_tiled(&tma_tensor_3, local_smem_q + i * TILE_M_1,
                             local_smem_bar_tma_q, i / BYTES_PER_ELEMENT_1,
                             row_coord, head_coord_1 + 0, batch_coord_1,
                             elect_one);
          }
        }
      } // if (kv_loop_left_bound < kv_loop_right_bound)
      wait_barrier(cast_smem_ptr_to_uint(&shared_storage.epilogue_done_mbar[0]),
                   epilogue_state);
      epilogue_state ^= 1;
      if (kv_loop_left_bound < kv_loop_right_bound) {
        int kv_loop = kv_loop_left_bound;
        // Load V
        const int v_stage = kv_mbar_state.index();
        const int v_phase = kv_mbar_state.phase();
        wait_barrier(
            cast_smem_ptr_to_uint(&shared_storage.tma_v_empty_mbar[v_stage]),
            v_phase);
        ++kv_mbar_state;

        if (true) {
          smem_bar_set_transaction_count(
              cast_smem_ptr_to_uint(&shared_storage.tma_v_full_mbar[v_stage]),
              vTmaTransactionBytes, elect_one);
        }
        {
          const int lds_page_buffer_state_idx = lds_page_buffer_state.index();
          uint32_t local_smem_v = cast_smem_ptr_to_uint(shared_storage.smem_V +
                                                        v_stage * vBufferElems);
          uint32_t local_smem_bar_tma_v =
              cast_smem_ptr_to_uint(&shared_storage.tma_v_full_mbar[v_stage]);
          const uint32_t p_col_coord = kv_loop * TILE_N_1;

          // global_load_shared_store_op 16 code
#pragma unroll
          for (int i = 0; i < TILE_O_1 * BYTES_PER_ELEMENT_1;
               i += 128 * CTA_MMA_1) {
            utmaldg_4d_tiled(&tma_tensor_1, local_smem_v + i * TILE_N_1,
                             local_smem_bar_tma_v, 0 + i / BYTES_PER_ELEMENT_1,
                             p_col_coord, head_coord_v_1, batch_coord_1,
                             elect_one);
          }
        }
      } // if (kv_loop_left_bound < kv_loop_right_bound)
      arrive_barrier(
          cast_smem_ptr_to_uint(&shared_storage.tmaldg_tile_started_mbar[0]));

// Mainloop to load K and V
#pragma unroll 1
      for (int kv_loop = kv_loop_left_bound + 1; kv_loop < kv_loop_right_bound;
           kv_loop += 1) {

        // Load K
        const int k_stage = kv_mbar_state.index();
        const int k_phase = kv_mbar_state.phase();
        wait_barrier(
            cast_smem_ptr_to_uint(&shared_storage.tma_k_empty_mbar[k_stage]),
            k_phase);

)KERNEL"
    R"KERNEL(
        if (true) {
          smem_bar_set_transaction_count(
              cast_smem_ptr_to_uint(&shared_storage.tma_k_full_mbar[k_stage]),
              kTmaTransactionBytes, elect_one);
        }
        {
          const int lds_page_buffer_state_idx = lds_page_buffer_state.index();
          uint32_t local_smem_k = cast_smem_ptr_to_uint(shared_storage.smem_K +
                                                        k_stage * kBufferElems);
          uint32_t local_smem_bar_tma_k =
              cast_smem_ptr_to_uint(&shared_storage.tma_k_full_mbar[k_stage]);
          const uint32_t p_col_coord = kv_loop * TILE_N_1;

          // global_load_shared_store_op 7 code
#pragma unroll
          for (int i = 0; i < TILE_O_1 * BYTES_PER_ELEMENT_1; i += 128) {
            utmaldg_4d_tiled(&tma_tensor_2, local_smem_k + i * TILE_N_1,
                             local_smem_bar_tma_k, i / BYTES_PER_ELEMENT_1,
                             p_col_coord, head_coord_k_1, batch_coord_1,
                             elect_one);
          }
        }
        if (kv_loop + 1 < kv_loop_right_bound) {
        }

        // Load V
        const int v_stage = kv_mbar_state.index();
        const int v_phase = kv_mbar_state.phase();
        wait_barrier(
            cast_smem_ptr_to_uint(&shared_storage.tma_v_empty_mbar[v_stage]),
            v_phase);
        ++kv_mbar_state;

        if (true) {
          smem_bar_set_transaction_count(
              cast_smem_ptr_to_uint(&shared_storage.tma_v_full_mbar[v_stage]),
              vTmaTransactionBytes, elect_one);
        }
        {
          const int lds_page_buffer_state_idx = lds_page_buffer_state.index();
          uint32_t local_smem_v = cast_smem_ptr_to_uint(shared_storage.smem_V +
                                                        v_stage * vBufferElems);
          uint32_t local_smem_bar_tma_v =
              cast_smem_ptr_to_uint(&shared_storage.tma_v_full_mbar[v_stage]);
          const uint32_t p_col_coord = kv_loop * TILE_N_1;

          // global_load_shared_store_op 16 code
#pragma unroll
          for (int i = 0; i < TILE_O_1 * BYTES_PER_ELEMENT_1;
               i += 128 * CTA_MMA_1) {
            utmaldg_4d_tiled(&tma_tensor_1, local_smem_v + i * TILE_N_1,
                             local_smem_bar_tma_v, 0 + i / BYTES_PER_ELEMENT_1,
                             p_col_coord, head_coord_v_1, batch_coord_1,
                             elect_one);
          }
        }

        ++lds_page_buffer_state;
      }
      if (kv_loop_left_bound < kv_loop_right_bound) {
        ++lds_page_buffer_state;
        ++sts_page_buffer_state;
      }
      __syncwarp();
      // Get next tile id
      wait_barrier(cast_smem_ptr_to_uint(
                       &shared_storage.scheduler_mbar[sched_state.index()]),
                   sched_state.phase());
      uint32_t tile_id_smem_int_ptr = cast_smem_ptr_to_uint(
          &shared_storage.tile_id[sched_state.index() * 8]);
      lds_128(tile_id_and_coord, tile_id_smem_int_ptr);
      ++sched_state;

      uint32_t tile_idx = tile_id_and_coord[0];
      uint32_t bidhb, l2_mod;
      uint32_t block = 0, bidhb_residual = 0;

      fastDivMod(l2_major_divmod, tile_idx, bidhb, l2_mod);
      if (bidhb < num_hb_quotient) {
        {
          fastDivMod(l2_minor_divmod, l2_mod, block, bidhb_residual);
        }
      } else {
        {
          fastDivMod(l2_minor_residual_divmod, l2_mod, block, bidhb_residual);
        }
      }

      int bidhb_actual = bidhb * l2_minor_divmod.val + bidhb_residual;
      uint32_t batch_idx, head_idx;
      fastDivMod(num_head_divmod, bidhb_actual, batch_idx, head_idx);
      block = num_block - 1 - block;

      lean_tile_id = tile_id_and_coord[0];

      blocked_row_coord = block + 0;
      head_coord_from_grid_1 = (head_idx) + 0;
      batch_coord_1 = batch_idx;
      is_valid_tile = tile_id_and_coord[2] & 1;

      head_coord_1 = head_coord_from_grid_1;
      head_coord_k_1 = head_coord_1 / attnDesc.q_heads_per_k;
      head_coord_v_1 = head_coord_1 / attnDesc.q_heads_per_v;

      q_row_coord = (blocked_row_coord * Tiles_Q * BMM1_TILE_M);
      int2 kv_loop_bounds = compute_kv_loop_bounds(
          q_row_coord, BMM1_TILE_M * Tiles_Q * CGA_M_1, TILE_N_1,
          actual_seqlen_kv_1, actual_seqlen_q_1, shift_right_bound, left_bound);
      kv_loop_left_bound = kv_loop_bounds.x;
      kv_loop_right_bound = kv_loop_bounds.y;
      if (actual_seqlen_q_1 == 0 || q_row_coord >= actual_seqlen_q_1) {
        kv_loop_right_bound = kv_loop_left_bound;
      }
    }
  }

  // thread_specialize_op 25 code
  else if (wid == SoftmaxWarps + CorrectionWarps + 2) {
    // TMASTG (epilogue)
    reg_dealloc<TmastgWarpRegs>();

    ElementO *smem_O_0 = shared_storage.smem_O;
    ElementO *smem_O_1 = shared_storage.smem_V;
    // Memory barrier states
    uint32_t epilogue_state = 0;

// Persistent loop over tiles
#pragma unroll 1
    while (is_valid_tile) {
      if (elect_one) {
        arrive_barrier(cast_smem_ptr_to_uint(
            &(shared_storage.read_tile_id_done_mbar[sched_state.index()])));
      }

      wait_barrier(
          cast_smem_ptr_to_uint(&shared_storage.tmaldg_tile_started_mbar[0]),
          epilogue_state);

      // global_store_op 20 decls

// TMASTG for sub-tile 0
#pragma unroll
      for (int i = 0; i < BMM2_TILE_N * sizeof(ElementO); i += 128) {
        wait_barrier(
            cast_smem_ptr_to_uint(&shared_storage.tma_o_0_full_mbar[i / 128]),
            epilogue_state);
        const uint32_t row_coord = q_row_coord;
        uint32_t smem_loc = cast_smem_ptr_to_uint(smem_O_0) + i * BMM2_TILE_M;
        if (actual_seqlen_q_1 > 0) {

          // global_store_op 20 code
          utmastg_4d_tiled(&tma_tensor_5, smem_loc, 0 + i / sizeof(ElementO),
                           row_coord, head_coord_1 + 0, batch_coord_1);
        }
      }

      tmastg_arrive();

// TMASTG for sub-tile 1
#pragma unroll
      for (int i = 0; i < BMM2_TILE_N * sizeof(ElementO); i += 128) {
        wait_barrier(
            cast_smem_ptr_to_uint(&shared_storage.tma_o_1_full_mbar[i / 128]),
            epilogue_state);
        const uint32_t row_coord = q_row_coord + TILE_M_1;
        uint32_t smem_loc = cast_smem_ptr_to_uint(smem_O_1) + i * BMM2_TILE_M;
        if (actual_seqlen_q_1 > 0) {

          // global_store_op 20 code
          utmastg_4d_tiled(&tma_tensor_5, smem_loc, 0 + i / sizeof(ElementO),
                           row_coord, head_coord_1 + 0, batch_coord_1);
        }
      }
      tmastg_arrive();

      tmastg_wait_count<1>();
      arrive_barrier(
          cast_smem_ptr_to_uint(&(shared_storage.tma_o_0_empty_mbar[0])));

      tmastg_wait_count<0>();
      arrive_barrier(
          cast_smem_ptr_to_uint(&(shared_storage.epilogue_done_mbar[0])));

      epilogue_state ^= 1;

      // Get next tile id
      wait_barrier(cast_smem_ptr_to_uint(
                       &shared_storage.scheduler_mbar[sched_state.index()]),
                   sched_state.phase());
      uint32_t tile_id_smem_int_ptr = cast_smem_ptr_to_uint(
          &shared_storage.tile_id[sched_state.index() * 8]);
      lds_128(tile_id_and_coord, tile_id_smem_int_ptr);
      ++sched_state;

      uint32_t tile_idx = tile_id_and_coord[0];
      uint32_t bidhb, l2_mod;
      uint32_t block = 0, bidhb_residual = 0;

      fastDivMod(l2_major_divmod, tile_idx, bidhb, l2_mod);
      if (bidhb < num_hb_quotient) {
        {
          fastDivMod(l2_minor_divmod, l2_mod, block, bidhb_residual);
        }
      } else {
        {
          fastDivMod(l2_minor_residual_divmod, l2_mod, block, bidhb_residual);
        }
      }

      int bidhb_actual = bidhb * l2_minor_divmod.val + bidhb_residual;
      uint32_t batch_idx, head_idx;
      fastDivMod(num_head_divmod, bidhb_actual, batch_idx, head_idx);
      block = num_block - 1 - block;

      lean_tile_id = tile_id_and_coord[0];

      blocked_row_coord = block + 0;
      head_coord_from_grid_1 = (head_idx) + 0;
      batch_coord_1 = batch_idx;
      is_valid_tile = tile_id_and_coord[2] & 1;

      head_coord_1 = head_coord_from_grid_1;
      head_coord_k_1 = head_coord_1 / attnDesc.q_heads_per_k;
      head_coord_v_1 = head_coord_1 / attnDesc.q_heads_per_v;

      q_row_coord = (blocked_row_coord * Tiles_Q * BMM1_TILE_M);
      int2 kv_loop_bounds = compute_kv_loop_bounds(
          q_row_coord, BMM1_TILE_M * Tiles_Q * CGA_M_1, TILE_N_1,
          actual_seqlen_kv_1, actual_seqlen_q_1, shift_right_bound, left_bound);
      kv_loop_left_bound = kv_loop_bounds.x;
      kv_loop_right_bound = kv_loop_bounds.y;
      if (actual_seqlen_q_1 == 0 || q_row_coord >= actual_seqlen_q_1) {
        kv_loop_right_bound = kv_loop_left_bound;
      }
    } // end of persistent loop
  }

  // thread_specialize_op 26 code
  else if (wid == SoftmaxWarps + CorrectionWarps + 3) {
    // Scheduler
    reg_dealloc<SchedulerWarpRegs>();

#pragma unroll 1
    while (is_valid_tile) {
      wait_barrier(
          cast_smem_ptr_to_uint(
              &shared_storage.read_tile_id_done_mbar[sched_state.index()]),
          sched_state.phase());
      uint32_t tile_id_smem_int_ptr = cast_smem_ptr_to_uint(
          &shared_storage.tile_id[sched_state.index() * 8]);

      smem_bar_set_transaction_count(
          cast_smem_ptr_to_uint(
              &shared_storage.scheduler_mbar[sched_state.index()]),
          16, elect_one);

      if (elect_one) {
        get_next_block_id(
            tile_id_smem_int_ptr,
            cast_smem_ptr_to_uint(
                &shared_storage.scheduler_mbar[sched_state.index()]));
      }
      __syncwarp();

      wait_barrier(cast_smem_ptr_to_uint(
                       &shared_storage.scheduler_mbar[sched_state.index()]),
                   sched_state.phase());
      lds_128(tile_id_and_coord, tile_id_smem_int_ptr);
      ++sched_state;

      is_valid_tile = tile_id_and_coord[2] & 1;
    }
  }
}
)KERNEL";
inline constexpr size_t sm100_d64_fprop_source_len = sizeof(sm100_d64_fprop_source) - 1;

inline constexpr const char sm100_d64_fprop_flags[] = R"FLAGS(
--gpu-architecture=sm_100a
--std=c++17
-w
--define-macro=__CUDACC_RTC__
-default-device
--use_fast_math
-Xptxas=-maxrregcount=128
-DCUDA_ENABLE_TENSOR_MEMORY_INTRINSICS=1
-DCUDA_ENABLE_VIRTCOUNT_INTRINSICS=1
-DCUDA_ENABLE_TMEM_MANAGEMENT_INTRINSICS=1
-DCUDA_BLACKWELL_TMA_SWIZZLE_ENABLED=1
-DCUDA_ENABLE_FLEXIBLE_CLUSTER=1
-DCUDA_PTX_TCMMA_V2_SUPPORTED=1
-DCUDA_PTX_TMEM_MANAGEMENT_SUPPORTED=1
-DCUDA_ENABLE_CLUSTER_MMA_INTRINSICS=1
-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1
-DCUTLASS_ENABLE_EXTENDED_PTX=1 
-DCUTLASS_ENABLE_INTERNAL_NVVM=1 
-DCUTLASS_CUDA_INTERNAL_L2_PREFETCH_ENABLED=1 
-DCUTLASS_CUDA_RP2RP_ENABLED=1 
-DCUTLASS_ENABLE_COMPILER_KNOBS=1 
-DCUTLASS_TEST_LEVEL=0 
-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 
-DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 
-DCUTLASS_DEBUG_TRACE_LEVEL=0 
-DCUTLASS_VERSIONS_GENERATED 
)FLAGS";
inline constexpr size_t sm100_d64_fprop_flags_len   = sizeof(sm100_d64_fprop_flags) - 1;

}  // namespace generated
}  // namespace experimental
}  // namespace cudnn_frontend
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#if defined(__clang__)
#pragma clang diagnostic pop
#endif