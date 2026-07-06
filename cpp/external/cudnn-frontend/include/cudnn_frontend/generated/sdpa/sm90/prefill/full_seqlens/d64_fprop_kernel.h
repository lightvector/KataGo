// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

// Suppress clang warning for overlength strings (concatenated result may exceed 65536)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverlength-strings"
#endif

// Suppress MSVC warning C4068 (unknown pragma) if clang pragmas are seen
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4068)
#endif

namespace cudnn_frontend {
namespace experimental {
namespace generated {

inline constexpr const char d64_fprop_source[] =
    R"KERNEL(
//receive_op 0 includes
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

#define CUDACC_VERSION __CUDACC_VER_MAJOR__ * 10 + __CUDACC_VER_MINOR__

namespace fort {
    static const uint64_t AMPERE_MEM_DESC_DEFAULT = uint64_t(0x1000000000000000ul);
    static const uint64_t MEM_DESC_DEFAULT = AMPERE_MEM_DESC_DEFAULT;
    static const uint32_t MMA_PEER_BIT_MASK = 0xFEFFFFFF;

#define NEG_INFINITY __int_as_float(0xff800000)

    typedef uint16_t half_t;
    typedef uint16_t bfloat16_t;

#define FORT_MIN(a,b) ((a) < (b) ? (a) : (b))
#define FORT_MAX(a,b) ((a) > (b) ? (a) : (b))
#define FORT_DIV_UP(a,b) (((a) + (b) - 1) / (b))
#define FORT_ROUND_UP(a,b) ((((a) + (b) - 1) / (b)) * (b))

    typedef struct tensor_descriptor {
        static const int MAX_DIMS = 12;

        int64_t num_dims;
        int64_t dims[MAX_DIMS];
        int64_t strides[MAX_DIMS];
    } tensor_descriptor;

    typedef struct FastDivisor {
        uint32_t val, shr, mul;
    } FastDivisor_t;

    inline __device__ void
    fastDivMod(const FastDivisor_t &d, uint32_t val, uint32_t &div, uint32_t &mod) {
        div = __umulhi((uint32_t)2 * val, d.mul) >> d.shr;
        mod = val - div * d.val;
    }
    __device__ __inline__ void cfence() {}

#if __CUDA_ARCH__ >= 900

    inline __device__ uint32_t elect_one_sync() {
        uint32_t pred = 0;
        uint32_t laneid = 0;
        asm volatile(
            "{\n"
            ".reg .b32 %%rx;\n"
            ".reg .pred %%px;\n"
            "     elect.sync %%rx|%%px, %2;\n"
            "@%%px mov.s32 %1, 1;\n"
            "     mov.s32 %0, %%rx;\n"
        "}\n" : "+r"(laneid), "+r"(pred) : "r"(0xFFFFFFFF));
        return pred;
    }

    template<int TARGET_REG_COUNT>
    inline __device__ void reg_alloc() {
        // const int TARGET_REG_COUNT = 232; // Example values (use with reg_delloc's VALUE together): 224, 232, 208, 216
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(TARGET_REG_COUNT));
    }

    template<int TARGET_REG_COUNT>
    inline __device__ void reg_dealloc() {
        // const int TARGET_REG_COUNT = 40; // Exmaple values: 56, 40, 88, 72
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(TARGET_REG_COUNT));
    }

    inline __device__ void named_barrier_arrive(uint32_t BARRIER_ID, uint32_t NUM_THREADS) {
        asm volatile("bar.arrive %0, %1;" : : "r"(BARRIER_ID), "r"(NUM_THREADS));
    }

    inline __device__ void named_barrier_wait(uint32_t BARRIER_ID, uint32_t NUM_THREADS) {
        asm volatile("bar.sync %0, %1;" :: "r"(BARRIER_ID), "r"(NUM_THREADS));
    }

    inline __device__ void tmastg_arrive() {
        asm volatile("cp.async.bulk.commit_group;");
    }

    inline __device__ void tmastg_wait() {
        asm volatile("cp.async.bulk.wait_group.read %0;" : :"n"(0):"memory");
    }

#endif

#if defined( __CUDA_ARCH__ ) && __CUDA_ARCH__ >= 900
    inline __device__ void warpgroup_arrive() {
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    }
    template <int N> inline __device__ void warpgroup_wait() {
        asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
    }
    inline __device__ void warpgroup_commit() {
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    }
    template <typename T>
    inline __device__ T &__wgmma_fence_operand(T &&reg) {
        asm volatile("" : "+r"(reg)::"memory");
        return reg;
    }
#endif

#if defined( __CUDA_ARCH__ ) && __CUDA_ARCH__ >= 900
    inline __device__ void smem_bar_init(uint32_t smem_ptr, uint32_t thread_count) {
        asm volatile ("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_ptr), "r"(thread_count));
    }

    inline __device__ void smem_bar_set_transaction_count(uint32_t smem_ptr, uint32_t expected_copy_bytes, uint32_t pred = 0) {
        asm volatile("{\n\t.reg .pred p;"
            " \n\tsetp.eq.u32 p, %2, 1;"
            " \n\t@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        " \n\t}" :: "r"(smem_ptr), "r"(expected_copy_bytes), "r"(pred));
    }

    inline __device__ uint32_t smem_bar_peek(uint32_t smem_ptr, uint32_t bar_phase) {
        uint32_t bar_phase_out;
        asm volatile("{\n\t.reg .pred       P1;"
            " \n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;"
        " \n\tselp.b32 %0, 1, 0, P1; \n\t}" :"=r" (bar_phase_out):"r"(smem_ptr),"r"(bar_phase));
        return bar_phase_out;
    }

    inline __device__ void smem_bar_wait(uint32_t smem_ptr, uint32_t bar_phase) {
        uint32_t large_val = 0x989680;
        asm volatile("{\n\t.reg .pred                P1;"
            " \n\tLAB_WAIT:"
            " \n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;"
            " \n\t@P1                       bra.uni DONE;"
            " \n\tbra.uni                   LAB_WAIT;"
            " \n\tDONE:"
        " \n\t}" :: "r"(smem_ptr),"r"(bar_phase), "r"(large_val));
    }

    inline __device__
    void
    wait_barrier(uint32_t smem_int_ptr,
    int phase_bit)                          // Current phase bit the barrier waiting to flip
    {
        asm volatile(
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(smem_int_ptr),
        "r"(phase_bit));
    }

    inline __device__ void smem_bar_arrive(uint32_t smem_ptr) {
        asm volatile("{\n\t.reg .b64 state;"
            " \n\tmbarrier.arrive.shared::cta.b64   state, [%0];"
        " \n\t}" :: "r"(smem_ptr));
    }

    // Barrier arrive on local smem
    inline __device__
    void arrive_barrier(uint32_t smem_addr) {
        asm volatile(
            "{\n\t"
            "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"
            "}"
            :
        : "r"(smem_addr));
    }

#endif

    extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *ptr);

    inline __device__ uint32_t get_smem_pointer(const void *ptr) {
        return __nvvm_get_smem_pointer(const_cast<void *>(ptr));
    }

    extern "C" __device__ inline void *__nv_cvta_shared_to_generic_impl(size_t __ptr) {
        return (void *)(void __attribute__((address_space(3))) *)__ptr;
    }
    inline __device__ void *set_smem_pointer(uint32_t ptr) {
        return __nv_cvta_shared_to_generic_impl(ptr);
    }

    inline __device__ uint32_t
    cast_smem_ptr_to_uint(void const* const ptr) {
        return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    }

    inline __device__ void stsm_x4(r32 dst, r32 src[4]) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(dst), "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]));
    }

    inline __device__ void fence_view_async_shared(void){
        asm volatile ("fence.proxy.async.shared::cta;\n");
    }

    inline __device__ void sts_32(r32 dst, r32 src[1]) {
        asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(dst), "r"(src[0]));
    }

    inline __device__ void lds_32(r32 dst[1], r32 src) {
        asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(dst[0]) : "r"(src));
    }

    inline __device__ void stg_32(void *ptr, r32 val[1]) {
        uint32_t *p = reinterpret_cast<uint32_t *>(ptr);
        p[0] = val[0];
    }

    // Forward ReLU activations

    inline __device__ uint32_t fp32_exp2(uint32_t in) {
        uint32_t out;
        asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=r"(out) : "r"(in));
        return out;
    }

    inline __device__ void fp32x2_to_bf16x2(r32 dst[1], const r32 src[2]) {
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(dst[0]) : "r"(src[1]), "r"(src[0]));
    }

    inline __device__ void bf16x2_to_fp32x2(r32 dst[2], const r32 src[1]) {
        asm volatile( \
            "{\n" \
            "    .reg .b16 lo, hi;\n" \
            "    mov.b32 {lo, hi}, %2;\n" \
            "    mov.b32 %0, {0, lo};\n" \
            "    mov.b32 %1, {0, hi};\n" \
        "}\n" : "=r"(dst[0]), "=r"(dst[1]) : "r"(src[0]));
    }

    //mma_pipeline_op 1 includes
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

    // CUtensorMap, when issued using bulk copy async functions, are cached in constant cache.
    // But the producer kernel may have directly written to global memory, without invalidating this constant cache.
    // This acquire fence, invalidates the memory address in constant cache, using UTMACCTL.IV.
    inline __device__ void tma_descriptor_fence_acquire(
    CUtensorMap const* p_desc
    ) {
#if (__CUDA_ARCH__ >= 900) && (CUDACC_VERSION >= 123)
        uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(p_desc);
        asm volatile (
            "fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n"
            :
            : "l"(gmem_int_desc)
        : "memory");
        asm volatile (
            "cvta.global.u64 %0, %0;\n"
            :
            : "l"(gmem_int_desc), "l"(gmem_int_desc)
        : "memory");
#endif
    }

    inline __device__ void utmaldg_4d_tiled (
    const void *p_desc,
    uint32_t urb0,  // smem offset
    uint32_t urb1,  // smem barrier offset
    int32_t urb2,   // m
    int32_t urb3,   // n
    int32_t urb4,   // b
    int32_t urb5,   // extra coord
    const uint32_t elect_one,
    const uint64_t mem_desc = MEM_DESC_DEFAULT
    ) {
        if (elect_one) {
#if __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
            asm volatile(
                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5}], [%6], %7;\n" ::
                "r"(urb0),
                "l"(reinterpret_cast<uint64_t>(p_desc)),
                "r"(urb2),
                "r"(urb3),
                "r"(urb4),
                "r"(urb5),
                "r"(urb1),
                "l"(mem_desc) : "memory"
            );
#else
            asm volatile(
                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5}], [%6], %7;\n" ::
                "r"(urb0),
                "l"(reinterpret_cast<uint64_t>(p_desc)),
                "r"(urb2),
                "r"(urb3),
                "r"(urb4),
                "r"(urb5),
                "r"(urb1),
                "l"(mem_desc) : "memory"
            );
#endif
        }
    }

    inline __device__ void utmastg_4d_tiled(
    const void *p_desc,
    uint32_t smem_offset,
    int32_t urb0,  // m
    int32_t urb1,  // n
    int32_t urb2,  // h
    int32_t urb3,  // b
    uint64_t mem_desc = MEM_DESC_DEFAULT
    ) {
        asm volatile(
            "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%1, %2, %3, %4}], [%5];\n" ::
            "l"(reinterpret_cast<uint64_t>(p_desc)),
            "r"(urb0),
            "r"(urb1),
            "r"(urb2),
            "r"(urb3),
            "r"(smem_offset) : "memory"
        );
    }

    inline __device__ void ldgdepbar() {
        asm volatile("cp.async.commit_group;\n" ::);
    }

    template<int STAGES>
    inline __device__ void depbar() {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES));
    }

    //mma_loop_op 2 includes

    //global_load_shared_store_op 4 includes

    // shared_load_op 5 includes

    // shared_load_op 6 includes

    //global_load_shared_store_op 7 includes

    //mma_op 8 includes
)KERNEL"
    R"KERNEL(
// === Section: GMMA Operations ===
    inline __device__ void bf16gmma_fp32_64x128x16_TN_asmem_bsmem_isb(const uint64_t &desc_a, const uint64_t &desc_b, r32 acc[64]) {
        __wgmma_mma_async_bf16(64, 128, 16, false, false, false, false, true, acc, desc_a, desc_b);
        warpgroup_commit();
    }

    // === Section: GMMA Operations ===
    inline __device__ void bf16gmma_fp32_64x128x16_TN_asmem_bsmem(const uint64_t &desc_a, const uint64_t &desc_b, r32 acc[64]) {
        __wgmma_mma_async_bf16(64, 128, 16, false, false, false, false, true, acc, desc_a, desc_b);
    }

#define BMM_S_GMMA     bf16gmma_fp32_64x128x16_TN_asmem_bsmem
#define BMM_S_GMMA_ISB bf16gmma_fp32_64x128x16_TN_asmem_bsmem_isb
#define BMM_S_GMMA_C0 bf16gmma_fp32_64x128x16_TN_asmem_bsmem_ignoreC

    //pointwise_calc_op 9 includes

    //global_load_op 10 includes

    //softmax_op 12 includes

    // global_store_op 13 includes

    // global_store_op 14 includes

    // shared_load_op 15 includes

    //global_load_shared_store_op 16 includes

    //mma_op 17 includes
    inline __device__ void bf16gmma_fp32_64x64x16_TT_arf_bsmem_isb(const r32 a[4], const uint64_t &desc_b, r32 acc[32]) {
        __wgmma_mma_async_bf16(64, 64, 16, false, false, true, false, true, acc, a, desc_b);
        warpgroup_commit();
    }

    inline __device__ void bf16gmma_fp32_64x64x16_TT_arf_bsmem(const r32 a[4], const uint64_t &desc_b, r32 acc[32]) {
        __wgmma_mma_async_bf16(64, 64, 16, false, false, true, false, true, acc, a, desc_b);
    }

#define BMM_O_GMMA     bf16gmma_fp32_64x64x16_TT_arf_bsmem
#define BMM_O_GMMA_ISB bf16gmma_fp32_64x64x16_TT_arf_bsmem_isb
#define BMM_O_GMMA_C0  bf16gmma_fp32_64x64x16_TT_arf_bsmem_ignoreC

    // global_store_op 20 includes

    //output_loop_op 3 includes

    // shared_store_op 18 includes

    // shared_load_op 19 includes
}  // namespace fort

//receive_op 0 types
using namespace fort;

static constexpr int THREADS_PER_WARP_0       = 32;
static constexpr int WARPS_PER_GROUP_0        = FORT_MIN(4, 384 / THREADS_PER_WARP_0);
static constexpr int THREADS_PER_WARP_GROUP_0 = THREADS_PER_WARP_0 * WARPS_PER_GROUP_0;
static constexpr int BITS_PER_REGISTER_0      = 32;  // NOTE: sizeof(uint)
static constexpr int BYTES_PER_REGISTER_0     = BITS_PER_REGISTER_0 / 8;
static constexpr int THREADS_PER_GROUP_0      = WARPS_PER_GROUP_0 * THREADS_PER_WARP_0;

// mma_pipeline_op 1 types
static constexpr int BYTES_PER_BANK_1    = 16;
static constexpr int BITS_PER_ELEMENT_1  = 16;
static constexpr int ELEMS_PER_BANK_1    = BYTES_PER_BANK_1 * 8 / BITS_PER_ELEMENT_1;
static constexpr int BYTES_PER_ELEMENT_1 = BITS_PER_ELEMENT_1 / 8;
static constexpr int BYTES_PER_FP16_1    = 2;
static constexpr int ELEMS_PER_VECTOR_1  = BYTES_PER_REGISTER_0 / BYTES_PER_ELEMENT_1;
static constexpr int BITS_PER_ACC_1      = 32;
static constexpr int BYTES_PER_ACC_1     = BITS_PER_ACC_1 / 8;
static constexpr int TILE_M_1            = 64;
static constexpr int TILE_N_1            = 128;
static constexpr int TILE_K_1            = 64;
static constexpr int TILE_O_1            = 64;
static constexpr int TILE_V_1            = TILE_K_1;
static constexpr int TILE_DV_1           = 64;
static constexpr int WARP_TILE_M_1       = 16;
static constexpr int WARP_TILE_N_1       = 16;
static constexpr int WARP_TILE_K_1       = 32 / BYTES_PER_ELEMENT_1;
static constexpr int WARPS_M_1           = 64 / 16;
static constexpr int WARPS_N_1           = 16 / 16;
static constexpr int WARPS_K_1           = 16 / 16;
static constexpr int STAGES_1            = 1;
static constexpr int GROUPS_M_1          = 1;
static constexpr int GMMA_TILE_M_1       = WARPS_M_1 * WARP_TILE_M_1;
static constexpr int GMMA_TILE_K_1       = WARPS_K_1 * WARP_TILE_K_1;
static constexpr int WARP_TILES_M_1      = TILE_M_1 / WARP_TILE_M_1;
static constexpr int WARP_TILES_N_1      = TILE_N_1 / WARP_TILE_N_1;
static constexpr int WARP_TILES_K_1      = TILE_K_1 / WARP_TILE_K_1;
static constexpr int WARP_REGS_1         = 8;
static constexpr int REGS_M_1            = WARP_TILES_M_1 / WARPS_M_1;
static constexpr int REGS_N_1            = WARP_TILES_N_1 / WARPS_N_1;
static constexpr int REGS_K_1            = WARP_TILES_K_1 / WARPS_K_1;
static constexpr int REGS_K_FOR_DP_1     = TILE_V_1 / 16;
static constexpr int REGS_dQ_1           = TILE_O_1 / 16;

static constexpr int REGS_O_1            = 4;

static constexpr int CGA_M_1             = 1;
static constexpr int CGA_N_1             = 1;
static constexpr int CTA_MMA_1           = 1;
static constexpr int M_TILES_PER_OUTPUT_TILE_1 = 4;
static constexpr int N_TILES_PER_OUTPUT_TILE_1 = 1;

static constexpr int CTA_TILE_M_1        = TILE_M_1 * GROUPS_M_1;
static constexpr int CTA_TILE_N_1        = 64;
static constexpr int CTA_TILE_K_1        = 128;
static constexpr int BUFFERS_Q_1         = 2;
static constexpr int BUFFERS_K_1         = 2;
static constexpr int BUFFERS_V_1         = 2;
static constexpr int BUFFERS_O_1         = 1;
static constexpr int BUFFERS_D_1         = 1;
static constexpr int SMEM_Q_1            = TILE_M_1 * TILE_K_1 * BYTES_PER_ELEMENT_1 * 2;
static constexpr int SMEM_K_1            = TILE_N_1 * TILE_K_1 * BYTES_PER_ELEMENT_1;
static constexpr int SMEM_V_1            = TILE_N_1 * TILE_K_1 * BYTES_PER_ELEMENT_1;
static constexpr int SMEM_PAGE_TABLE_1   = FORT_ROUND_UP(TILE_N_1 * 2 * 2 * sizeof(int), 1024);
static constexpr int SMEM_OFFSET_Q_1     = 1024 + (false ? 0 : SMEM_PAGE_TABLE_1);
static constexpr int SMEM_OFFSET_K_1     = SMEM_OFFSET_Q_1 + SMEM_Q_1 * BUFFERS_Q_1;
static constexpr int SMEM_OFFSET_V_1     = SMEM_OFFSET_K_1 + SMEM_K_1 * BUFFERS_K_1;
static constexpr int SMEM_OFFSET_O_1     = SMEM_OFFSET_V_1 + SMEM_V_1 * BUFFERS_V_1;
static constexpr int SMEM_OFFSET_D_1     = SMEM_OFFSET_O_1;
static constexpr int SMEM_OFFSET_BAND_BIAS_1 = SMEM_OFFSET_O_1;

static constexpr int THREADS_ON_TILE_BARRIER_1 = 256+32+0;
static constexpr int TILE_ID_SYNC_2_BARRIER_1  = 2;
static constexpr int TILE_ID_SYNC_3_BARRIER_1  = 3;
static constexpr int SOFTMAX_1_BARRIER_1       = 4;
static constexpr int SOFTMAX_2_BARRIER_1       = 5;
static constexpr int MATH_WORKGROUP_1          = 6;
static constexpr int TMA_SYNC_BARRIER_1        = 7;

static constexpr uint32_t BYTES_PER_GMMA_K_1               = GMMA_TILE_K_1 * BYTES_PER_ELEMENT_1;
static constexpr uint32_t BYTES_PER_GMMA_K_NO_4LSB_1       = BYTES_PER_GMMA_K_1 >> 4;
static constexpr uint32_t BYTES_PER_GMMA_K_TRANS_1         = GMMA_TILE_K_1 * 128;
static constexpr uint32_t BYTES_PER_GMMA_K_NO_4LSB_TRANS_1 = BYTES_PER_GMMA_K_TRANS_1 >> 4;

static constexpr uint32_t GMMA_DESCRIPTOR_SWIZZLE_MODE_BITS_1              = 2;
static constexpr uint32_t GMMA_DESCRIPTOR_SWIZZLE_MODE_SHIFT_1             = 62;  // bits 63-62
static constexpr uint64_t GMMA_DESCRIPTOR_SWIZZLE_MODE_VALUE_Q_1           = 1;  // SWIZZLE_NONE=0, SWIZZLE_128B=1, SWIZZLE_64B=2, SWIZZLE_32B=3
static constexpr uint64_t GMMA_DESCRIPTOR_SWIZZLE_MODE_IN_BIT_LOCATION_Q_1 = (GMMA_DESCRIPTOR_SWIZZLE_MODE_VALUE_Q_1 &
((1u << GMMA_DESCRIPTOR_SWIZZLE_MODE_BITS_1) - 1))
<< GMMA_DESCRIPTOR_SWIZZLE_MODE_SHIFT_1;
static constexpr uint64_t GMMA_DESCRIPTOR_SWIZZLE_MODE_VALUE_K_1           = 1;  // SWIZZLE_NONE=0, SWIZZLE_128B=1, SWIZZLE_64B=2, SWIZZLE_32B=3
static constexpr uint64_t GMMA_DESCRIPTOR_SWIZZLE_MODE_IN_BIT_LOCATION_K_1 = (GMMA_DESCRIPTOR_SWIZZLE_MODE_VALUE_K_1 &
((1u << GMMA_DESCRIPTOR_SWIZZLE_MODE_BITS_1) - 1))
<< GMMA_DESCRIPTOR_SWIZZLE_MODE_SHIFT_1;
static constexpr uint64_t GMMA_DESCRIPTOR_SWIZZLE_MODE_VALUE_V_1           = 1;  // SWIZZLE_NONE=0, SWIZZLE_128B=1, SWIZZLE_64B=2, SWIZZLE_32B=3
static constexpr uint64_t GMMA_DESCRIPTOR_SWIZZLE_MODE_IN_BIT_LOCATION_V_1 = (GMMA_DESCRIPTOR_SWIZZLE_MODE_VALUE_V_1 &
((1u << GMMA_DESCRIPTOR_SWIZZLE_MODE_BITS_1) - 1))
<< GMMA_DESCRIPTOR_SWIZZLE_MODE_SHIFT_1;

static constexpr uint32_t GMMA_DESCRIPTOR_STRIDE_BYTE_OFFSET_SHIFT_1             = 32;  // bits 45-32
static constexpr uint64_t BYTES_PER_LEADING_DIM_1                                = 128;
static constexpr uint64_t STRIDE_BYTE_OFFSET_Q_1                                 = (BYTES_PER_LEADING_DIM_1 * 8) / 16;  // SWIZZLE_NONE=0, SWIZZLE_128B=8, SWIZZLE_64B=4, SWIZZLE_32B=2
static constexpr uint64_t GMMA_DESCRIPTOR_STRIDE_BYTE_OFFSET_IN_BIT_LOCATION_Q_1 = STRIDE_BYTE_OFFSET_Q_1 << GMMA_DESCRIPTOR_STRIDE_BYTE_OFFSET_SHIFT_1;
static constexpr uint64_t STRIDE_BYTE_OFFSET_K_1                                 = (BYTES_PER_LEADING_DIM_1 * 8) / 16;  // SWIZZLE_NONE=0, SWIZZLE_128B=8, SWIZZLE_64B=4, SWIZZLE_32B=2
static constexpr uint64_t GMMA_DESCRIPTOR_STRIDE_BYTE_OFFSET_IN_BIT_LOCATION_K_1 = STRIDE_BYTE_OFFSET_K_1 << GMMA_DESCRIPTOR_STRIDE_BYTE_OFFSET_SHIFT_1;
static constexpr uint64_t STRIDE_BYTE_OFFSET_V_1                                 = (BYTES_PER_LEADING_DIM_1 * 8) / 16;  // SWIZZLE_NONE=0, SWIZZLE_128B=8, SWIZZLE_64B=4, SWIZZLE_32B=2
static constexpr uint64_t GMMA_DESCRIPTOR_STRIDE_BYTE_OFFSET_IN_BIT_LOCATION_V_1 = STRIDE_BYTE_OFFSET_V_1 << GMMA_DESCRIPTOR_STRIDE_BYTE_OFFSET_SHIFT_1;

static constexpr uint32_t GMMA_DESCRIPTOR_LD_BYTE_OFFSET_SHIFT_1             = 16;  // bits 29-16
static constexpr uint32_t GMMA_DESCRIPTOR_LD_BYTE_OFFSET_VALUE_Q_1           = 128 * 64 / 16;  // Not used???
static constexpr uint64_t GMMA_DESCRIPTOR_LD_BYTE_OFFSET_IN_BIT_LOCATION_Q_1 = GMMA_DESCRIPTOR_LD_BYTE_OFFSET_VALUE_Q_1
<< GMMA_DESCRIPTOR_LD_BYTE_OFFSET_SHIFT_1;
static constexpr uint32_t GMMA_DESCRIPTOR_LD_BYTE_OFFSET_VALUE_K_1           = 128 * 32 / 16;  // Not used???
static constexpr uint64_t GMMA_DESCRIPTOR_LD_BYTE_OFFSET_IN_BIT_LOCATION_K_1 = GMMA_DESCRIPTOR_LD_BYTE_OFFSET_VALUE_K_1
<< GMMA_DESCRIPTOR_LD_BYTE_OFFSET_SHIFT_1;
static constexpr uint32_t GMMA_DESCRIPTOR_LD_BYTE_OFFSET_VALUE_V_1           = 128 * TILE_N_1 / 16;
static constexpr uint64_t GMMA_DESCRIPTOR_LD_BYTE_OFFSET_IN_BIT_LOCATION_V_1 = GMMA_DESCRIPTOR_LD_BYTE_OFFSET_VALUE_V_1
<< GMMA_DESCRIPTOR_LD_BYTE_OFFSET_SHIFT_1;

inline __device__ constexpr uint64_t create_gmma_desc_q_1() {
    return ((uint64_t)0 | GMMA_DESCRIPTOR_SWIZZLE_MODE_IN_BIT_LOCATION_Q_1)
    | GMMA_DESCRIPTOR_STRIDE_BYTE_OFFSET_IN_BIT_LOCATION_Q_1
    | GMMA_DESCRIPTOR_LD_BYTE_OFFSET_IN_BIT_LOCATION_Q_1;
}

inline __device__ constexpr uint64_t create_gmma_desc_k_1() {
    return ((uint64_t)0 | GMMA_DESCRIPTOR_SWIZZLE_MODE_IN_BIT_LOCATION_K_1)
    | GMMA_DESCRIPTOR_STRIDE_BYTE_OFFSET_IN_BIT_LOCATION_K_1
    | GMMA_DESCRIPTOR_LD_BYTE_OFFSET_IN_BIT_LOCATION_K_1;
}

// === Section: GMMA Descriptors & Mask Helpers ===
)KERNEL"
    R"KERNEL(
// === Section: GMMA Descriptors & Mask Helpers ===
inline __device__ constexpr uint64_t create_gmma_desc_v_1() {
    return ((uint64_t)0 | GMMA_DESCRIPTOR_SWIZZLE_MODE_IN_BIT_LOCATION_V_1)
    | GMMA_DESCRIPTOR_STRIDE_BYTE_OFFSET_IN_BIT_LOCATION_V_1
    | GMMA_DESCRIPTOR_LD_BYTE_OFFSET_IN_BIT_LOCATION_V_1;
}

class Gmma_descriptor {
    public:
    inline __device__ Gmma_descriptor(uint64_t desc_) {
        desc = desc_;
    }
    template<int BYTES_PER_BUFFER, int BUFFER_COUNT>
    inline __device__ void set_smem(uint32_t smem) {
        desc |= (static_cast<uint64_t>(smem & 0xFFFFFF) >> 4);

        max_desc = desc + (BYTES_PER_BUFFER >> 4) * (BUFFER_COUNT - 1);
    }
    template<int BYTES_PER_BUFFER, int BUFFER_COUNT>
    inline __device__ void increment_smem_buffer() {
        int2 &tmp = reinterpret_cast<int2 &>(desc);
        tmp.x += (desc >= max_desc) ? -(BYTES_PER_BUFFER >> 4) * (BUFFER_COUNT - 1)
        :  (BYTES_PER_BUFFER >> 4);
    }

    uint64_t desc;
    uint64_t max_desc;
};

//global_load_shared_store_op 4 types
static constexpr int BITS_PER_ELEMENT_4  = 16;
static constexpr int BYTES_PER_ELEMENT_4 = BITS_PER_ELEMENT_4 / 8;
static constexpr int BYTES_PER_ACCESS_4  = 16;
static constexpr int TILE_M_4            = 64;
static constexpr int TILE_N_4            = 64;
static constexpr int BYTES_PER_SMEM_4    = TILE_M_4 * TILE_N_4 * BYTES_PER_ELEMENT_4;

// shared_load_op 5 types
static constexpr int BITS_PER_ELEMENT_5        = 16;
static constexpr int BYTES_PER_ELEMENT_5       = 2;
static constexpr int BYTES_PER_QUAD_5          = 16;
static constexpr int WARP_TILE_M_5             = 16;
static constexpr int WARP_TILE_N_5             = 16;
static constexpr int TILE_M_5                  = 64;
static constexpr int TILE_N_5                  = 64;
static constexpr int BYTES_PER_SMEM_5          = TILE_M_5 * TILE_N_5 * BYTES_PER_ELEMENT_5;
static constexpr int WARPS_M_5                 = 4;
static constexpr int WARPS_N_5                 = 1;
static constexpr int WARP_TILES_M_5            = TILE_M_5 / WARP_TILE_M_5;
static constexpr int WARP_TILES_N_5            = TILE_N_5 / WARP_TILE_N_5;
static constexpr int THREADS_PER_WARP_TILE_M_5 = 8;
static constexpr int THREADS_PER_WARP_TILE_N_5 = 4;
static constexpr int REGS_M_5                  = WARP_TILES_M_5 / WARPS_M_5;
static constexpr int REGS_N_5                  = WARP_TILES_N_5 / WARPS_N_5;
static constexpr int BYTES_PER_LD_5            = 128;
static constexpr int SWIZZLE_SCALE_5           = FORT_MIN(BYTES_PER_LD_5 / 16, WARPS_PER_GROUP_0 * 4);

// shared_load_op 6 types
static constexpr int BITS_PER_ELEMENT_6        = 16;
static constexpr int BYTES_PER_ELEMENT_6       = 2;
static constexpr int BYTES_PER_QUAD_6          = 16;
static constexpr int WARP_TILE_M_6             = 16;
static constexpr int WARP_TILE_N_6             = 16;
static constexpr int TILE_M_6                  = 128;
static constexpr int TILE_N_6                  = 64;

static constexpr int BYTES_PER_SMEM_6          = TILE_M_6 * TILE_N_6 * BYTES_PER_ELEMENT_6;
static constexpr int WARPS_M_6                 = 1;
static constexpr int WARPS_N_6                 = 1;
static constexpr int WARP_TILES_M_6            = TILE_M_6 / WARP_TILE_M_6;
static constexpr int WARP_TILES_N_6            = TILE_N_6 / WARP_TILE_N_6;
static constexpr int THREADS_PER_WARP_TILE_M_6 = 8;
static constexpr int THREADS_PER_WARP_TILE_N_6 = 4;
static constexpr int REGS_M_6                  = WARP_TILES_M_6 / WARPS_M_6;
static constexpr int REGS_N_6                  = WARP_TILES_N_6 / WARPS_N_6;
static constexpr int BYTES_PER_LD_6            = 128;
static constexpr int SWIZZLE_SCALE_6           = FORT_MIN(BYTES_PER_LD_6 / 16, WARPS_PER_GROUP_0 * 4);

//global_load_shared_store_op 7 types
static constexpr int BITS_PER_ELEMENT_7  = 16;
static constexpr int BYTES_PER_ELEMENT_7 = BITS_PER_ELEMENT_7 / 8;
static constexpr int BYTES_PER_ACCESS_7  = 16;
static constexpr int TILE_M_7            = 128;
static constexpr int TILE_N_7            = 64;
static constexpr int BYTES_PER_SMEM_7    = TILE_M_7 * TILE_N_7 * BYTES_PER_ELEMENT_7;

//mma_op 8 types
static constexpr int BYTES_PER_ELEMENT_8 = 2;
static constexpr int BYTES_PER_ACC_8     = 4;
static constexpr int WARP_TILE_M_8       = 16;
static constexpr int WARP_TILE_N_8       = 16;
static constexpr int WARP_TILE_K_8       = 16;
static constexpr int TILE_M_8            = 64;
static constexpr int TILE_N_8            = 128;
static constexpr int TILE_K_8            = 64;
static constexpr int WARPS_M_8           = 4;
static constexpr int WARPS_N_8           = 1;
static constexpr int WARP_TILES_M_8      = TILE_M_8 / WARP_TILE_M_8;
static constexpr int WARP_TILES_N_8      = TILE_N_8 / WARP_TILE_N_8;
static constexpr int MMA_STEPS_K_8       = TILE_K_8 / WARP_TILE_K_8;
static constexpr int WARP_REGS_8         = 8;
static constexpr int REGS_M_8            = WARP_TILES_M_8 / WARPS_M_8;
static constexpr int REGS_N_8            = WARP_TILES_N_8 / WARPS_N_8;

//pointwise_calc_op 9 types
static constexpr int WARP_TILE_M_9       = 16;
static constexpr int WARP_TILE_N_9       = 16;
static constexpr int TILE_M_9            = 64;
static constexpr int TILE_N_9            = 128;
static constexpr int WARPS_M_9           = 4;
static constexpr int WARPS_N_9           = 1;
static constexpr int WARP_TILES_M_9      = TILE_M_9 / WARP_TILE_M_9;
static constexpr int WARP_TILES_N_9      = TILE_N_9 / WARP_TILE_N_9;
static constexpr int WARP_REGS_9         = 8;
static constexpr int REGS_M_9            = WARP_TILES_M_9 / WARPS_M_9;
static constexpr int REGS_N_9            = WARP_TILES_N_9 / WARPS_N_9;

//global_load_op 10 types
static constexpr int BITS_PER_ELEMENT_10     = 32;
static constexpr int BYTES_PER_ELEMENT_10    = 4;
static constexpr int REGISTERS_PER_VECTOR_10 = 1;
static constexpr int REGISTERS_PER_ACCESS_10 = 1;

//mha_mask_op 11 types
static constexpr int WARP_TILE_M_11             = 16;
static constexpr int WARP_TILE_N_11             = 16;
static constexpr int TILE_M_11                  = 64;
static constexpr int TILE_N_11                  = 128;
static constexpr int WARPS_M_11                 = 4;
static constexpr int WARPS_N_11                 = 1;
static constexpr int WARP_TILES_M_11            = TILE_M_11 / WARP_TILE_M_11;
static constexpr int WARP_TILES_N_11            = TILE_N_11 / WARP_TILE_N_11;
static constexpr int WARP_REGS_11               = 8;
static constexpr int REGS_M_11                  = WARP_TILES_M_11 / WARPS_M_11;
static constexpr int REGS_N_11                  = WARP_TILES_N_11 / WARPS_N_11;
inline __device__ bool compute_diagonal_band_mask_11(const int row, const int col, const int actual_seqlen_kv, const int actual_seqlen_q, const int shift_right_bound, const int left_bound) {
    constexpr bool is_bottom_right_alignment = false;
    constexpr bool is_shift_right_bound = false;
    constexpr bool is_right_bound = true;
    constexpr bool is_left_bound = false;
    constexpr bool need_oob_check = false;

    const int diag = is_bottom_right_alignment ? row + (actual_seqlen_kv - actual_seqlen_q) : row;
    const int shifted_right_bound = is_shift_right_bound ? diag + shift_right_bound : diag;
    const bool right_bound_mask = is_right_bound ? col <= shifted_right_bound : true;
    const bool left_bound_mask = is_left_bound ? col + left_bound > diag : true;
    const bool oob_check = need_oob_check ? (row < actual_seqlen_q && col < actual_seqlen_kv) : true;
    return right_bound_mask && left_bound_mask && oob_check;
}

//softmax_op 12 types
static constexpr int BYTES_PER_ELEMENT_12       = 2;
static constexpr int WARP_TILE_M_12             = 16;
static constexpr int WARP_TILE_N_12             = 16;
static constexpr int TILE_M_12                  = 64;
static constexpr int TILE_N_12                  = 128;
static constexpr int WARPS_M_12                 = 4;
static constexpr int WARPS_N_12                 = 1;
static constexpr int WARP_TILES_M_12            = TILE_M_12 / WARP_TILE_M_12;
static constexpr int WARP_TILES_N_12            = TILE_N_12 / WARP_TILE_N_12;
static constexpr int THREADS_PER_WARP_TILE_M_12 = 8;

static constexpr int THREADS_PER_WARP_TILE_N_12 = 4;
static constexpr int WARP_REGS_12               = 8;
static constexpr int REGS_M_12                  = WARP_TILES_M_12 / WARPS_M_12;
static constexpr int REGS_N_12                  = WARP_TILES_N_12 / WARPS_N_12;
static constexpr int ROWS_PER_THREAD_12         = TILE_M_12 / WARPS_M_12 / THREADS_PER_WARP_TILE_M_12;

// global_store_op 13 types
static constexpr int BYTES_PER_ELEMENT_13       = 4;
static constexpr int WARP_TILE_M_13             = 16;
static constexpr int WARP_TILE_N_13             = 16;
static constexpr int TILE_M_13                  = 64;
static constexpr int TILE_N_13                  = 128;
static constexpr int WARPS_M_13                 = 4;
static constexpr int WARPS_N_13                 = 1;
static constexpr int WARP_TILES_M_13            = TILE_M_13 / WARP_TILE_M_13;
static constexpr int WARP_TILES_N_13            = TILE_N_13 / WARP_TILE_N_13;
static constexpr int THREADS_PER_WARP_TILE_M_13 = 8;
static constexpr int THREADS_PER_WARP_TILE_N_13 = 4;
static constexpr int ROWS_PER_THREAD_13         = TILE_M_13 / WARPS_M_13 / THREADS_PER_WARP_TILE_M_13;

// global_store_op 14 types
static constexpr int BYTES_PER_ELEMENT_14       = 4;
static constexpr int WARP_TILE_M_14             = 16;
static constexpr int WARP_TILE_N_14             = 16;
static constexpr int TILE_M_14                  = 64;
static constexpr int TILE_N_14                  = 128;
static constexpr int WARPS_M_14                 = 4;
static constexpr int WARPS_N_14                 = 1;
static constexpr int WARP_TILES_M_14            = TILE_M_14 / WARP_TILE_M_14;
static constexpr int WARP_TILES_N_14            = TILE_N_14 / WARP_TILE_N_14;
static constexpr int THREADS_PER_WARP_TILE_M_14 = 8;
static constexpr int THREADS_PER_WARP_TILE_N_14 = 4;
static constexpr int ROWS_PER_THREAD_14         = TILE_M_14 / WARPS_M_14 / THREADS_PER_WARP_TILE_M_14;

// shared_load_op 15 types
static constexpr int BITS_PER_ELEMENT_15        = 16;
static constexpr int BYTES_PER_ELEMENT_15       = 2;
static constexpr int BYTES_PER_QUAD_15          = 16;
static constexpr int WARP_TILE_M_15             = 16;
static constexpr int WARP_TILE_N_15             = 16;
static constexpr int TILE_M_15                  = 128;
static constexpr int TILE_N_15                  = 64;
static constexpr int BYTES_PER_SMEM_15          = TILE_M_15 * TILE_N_15 * BYTES_PER_ELEMENT_15;
static constexpr int WARPS_M_15                 = 1;
static constexpr int WARPS_N_15                 = 1;
static constexpr int WARP_TILES_M_15            = TILE_M_15 / WARP_TILE_M_15;
static constexpr int WARP_TILES_N_15            = TILE_N_15 / WARP_TILE_N_15;
static constexpr int THREADS_PER_WARP_TILE_M_15 = 8;
static constexpr int THREADS_PER_WARP_TILE_N_15 = 4;
static constexpr int REGS_M_15                  = WARP_TILES_M_15 / WARPS_M_15;
static constexpr int REGS_N_15                  = WARP_TILES_N_15 / WARPS_N_15;
static constexpr int BYTES_PER_LD_15            = 128;
static constexpr int SWIZZLE_SCALE_15           = FORT_MIN(BYTES_PER_LD_15 / 16, WARPS_PER_GROUP_0 * 4);

//global_load_shared_store_op 16 types
static constexpr int BITS_PER_ELEMENT_16  = 16;
static constexpr int BYTES_PER_ELEMENT_16 = BITS_PER_ELEMENT_16 / 8;
static constexpr int BYTES_PER_ACCESS_16  = 16;
static constexpr int TILE_M_16            = 128;
static constexpr int TILE_N_16            = 64;
static constexpr int BYTES_PER_SMEM_16    = TILE_M_16 * TILE_N_16 * BYTES_PER_ELEMENT_16;

//mma_op 17 types
static constexpr int BYTES_PER_ELEMENT_17 = 2;
static constexpr int BYTES_PER_ACC_17     = 4;
static constexpr int WARP_TILE_M_17       = 16;
static constexpr int WARP_TILE_N_17       = 16;
static constexpr int WARP_TILE_K_17       = 16;
static constexpr int TILE_M_17            = 64;
static constexpr int TILE_N_17            = 64;
static constexpr int TILE_K_17            = 128;
static constexpr int WARPS_M_17           = 4;
static constexpr int WARPS_N_17           = 1;
static constexpr int WARP_TILES_M_17      = TILE_M_17 / WARP_TILE_M_17;
static constexpr int WARP_TILES_N_17      = TILE_N_17 / WARP_TILE_N_17;
static constexpr int MMA_STEPS_K_17       = TILE_K_17 / WARP_TILE_K_17;
static constexpr int WARP_REGS_17         = 8;
static constexpr int REGS_M_17            = WARP_TILES_M_17 / WARPS_M_17;
static constexpr int REGS_N_17            = WARP_TILES_N_17 / WARPS_N_17;

// global_store_op 20 types

// mma_loop_op 2 types
inline __device__ int2 compute_kv_loop_bounds(const int row_coord, const int ROW_TILE_SIZE, const int COL_TILE_SIZE, const int actual_seqlen_kv, const int actual_seqlen_q, const int shift_right_bound, const int left_bound)
{
    constexpr bool is_right_bound = true;
    constexpr bool is_right_bound_bottom_right_alignment = false;
    constexpr bool is_shift_right_bound = false;
    constexpr bool is_left_bound = false;
    constexpr bool is_left_bound_bottom_right_alignment = false;

    const int right_bound_diagonal = is_right_bound_bottom_right_alignment ? row_coord + (actual_seqlen_kv - actual_seqlen_q) : row_coord;
    const int shifted_right_bound_diagonal = is_shift_right_bound ? right_bound_diagonal + shift_right_bound : right_bound_diagonal;
    const int left_bound_diagonal = is_left_bound_bottom_right_alignment ? row_coord + (actual_seqlen_kv - actual_seqlen_q) : row_coord;

    const int kv_loop_left_bound = is_left_bound ? FORT_MAX(0, (left_bound_diagonal - left_bound) / COL_TILE_SIZE) : 0;
    const int kv_loop_right_bound = is_right_bound ? FORT_MIN(FORT_DIV_UP(shifted_right_bound_diagonal + ROW_TILE_SIZE, COL_TILE_SIZE), FORT_DIV_UP(actual_seqlen_kv, COL_TILE_SIZE)) : FORT_DIV_UP(actual_seqlen_kv, COL_TILE_SIZE);
    return make_int2(kv_loop_left_bound, kv_loop_right_bound);
}

// output_loop_op 3 types
static constexpr int BYTES_PER_BANK_3        = 16;
static constexpr int LDS_TILE_M_3            = 16;
static constexpr int LDS_TILE_N_3            = 32;
static constexpr int ACC_CORE_MATRIX_ROWS_3  = 8;  // NOTE: fixed due to GMMA design
static constexpr int ACC_CORE_MATRIX_COLS_3  = 8;  // NOTE: fixed due to GMMA design
static constexpr int ELEMENTS_PER_VECTOR_3   = 8;
static constexpr int LDS_PER_TILE_3          = ELEMENTS_PER_VECTOR_3 / 2;
static constexpr int LDS_TILES_N_3           = FORT_MAX(CTA_TILE_N_1 / LDS_TILE_N_3, 1);
static constexpr int LDS_TILES_M_3           = FORT_MAX(1, (CTA_TILE_M_1 / GROUPS_M_1) / (LDS_TILE_M_3 * WARPS_PER_GROUP_0));
static constexpr int VECTORS_PER_LDS_TILES_3 = LDS_TILE_M_3 / ACC_CORE_MATRIX_ROWS_3;
static constexpr int STS_PER_OUTPUT_TILE_N_3 = CTA_TILE_N_1 / ACC_CORE_MATRIX_COLS_3;
static constexpr int STSM_X4_PER_OUTPUT_TILE_N_3 = CTA_TILE_N_1 / ACC_CORE_MATRIX_COLS_3 / 2;
static constexpr int BANKS_PER_PAD_LINE_3    = BYTES_PER_ACC_1 == 4 ? 1 : 2;

static constexpr int PADDING_BYTES_3         = BANKS_PER_PAD_LINE_3 * BYTES_PER_BANK_3 * LDS_PER_TILE_3;
)KERNEL"
    R"KERNEL(
static constexpr int BYTES_PER_STS_PER_WARP_3= 16 * THREADS_PER_WARP_0;  // 16 bytes due to STS_128
static constexpr int BYTES_PER_LDS_TILE_3    = LDS_TILE_N_3 * LDS_TILE_M_3 * BYTES_PER_ACC_1 + PADDING_BYTES_3;
static constexpr int BYTES_PER_WARP_3        = LDS_TILES_N_3 * BYTES_PER_LDS_TILE_3;

static constexpr int ELEMS_PER_STS_BLOCK_3 = 16;

static constexpr int STG_THREADS_PER_TILE_N_3        = LDS_TILE_N_3 / ELEMENTS_PER_VECTOR_3;
static constexpr int EPILOGUE_SMEM_SIZE_PER_XMMA_M_3 = BYTES_PER_WARP_3 * WARPS_PER_GROUP_0;

// shared_store_op 18 types

// shared_load_op 19 types
static constexpr int BITS_PER_ELEMENT_19     = 32 * 1;
static constexpr int BYTES_PER_ELEMENT_19    = BITS_PER_ELEMENT_19 / 8;
static constexpr int BITS_PER_VECTOR_19      = BITS_PER_ELEMENT_19 * ELEMENTS_PER_VECTOR_3;
static constexpr int BYTES_PER_VECTOR_19     = BITS_PER_VECTOR_19 / 8;
static constexpr int REGISTERS_PER_VECTOR_19 = BITS_PER_VECTOR_19 / BITS_PER_REGISTER_0;

//receive_op 0 code

// === Section: Kernel Function ===
)KERNEL"
    R"KERNEL(
// === Section: Kernel Function ===
extern "C" __global__ __launch_bounds__(384, 1)
void cudnn_generated_oss_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x64_4x1x1_cga1x1x1_kernel0_0(const AttentionDescriptor_t attnDesc
, int num_tiles_1
, const FastDivisor_t tiles_hr_div_1
, const FastDivisor_t tiles_r_div_1
, int32_t* tile_id_counter
, __grid_constant__ const CUtensorMap tma_Q
, __grid_constant__ const CUtensorMap tma_K
, float attn_scale
, float neg_infinity
, void* d_max
, fort::tensor_descriptor desc_max
, void* d_sum_exp
, fort::tensor_descriptor desc_sum_exp
, __grid_constant__ const CUtensorMap tma_V
, __grid_constant__ const CUtensorMap tma_O
) {
    extern __shared__ char smem_[];
    uint32_t smem_0 = get_smem_pointer(smem_);

    //receive_op 0 decls

    // mma_pipeline_op 1 decls
    const uint32_t tid = threadIdx.x % THREADS_PER_GROUP_0;
    const uint32_t gid = threadIdx.x / THREADS_PER_GROUP_0;
    const uint32_t wid = threadIdx.x / 32;
    const uint32_t tiw = threadIdx.x % 32;

    const uint32_t elect_one = fort::elect_one_sync();

    uint32_t smem_1         = smem_0;
    uint32_t smem_bar_qkv_1 = smem_1;
    uint32_t smem_page           = smem_1 + 1024;
    uint32_t smem_Q       = smem_1 + SMEM_OFFSET_Q_1;
    uint32_t smem_K       = smem_1 + SMEM_OFFSET_K_1;
    uint32_t smem_V       = smem_1 + SMEM_OFFSET_V_1;
    uint32_t smem_O       = smem_1 + SMEM_OFFSET_O_1;
    uint32_t smem_d_1       = smem_1 + SMEM_OFFSET_D_1;
    uint32_t smem_band_bias_1 = smem_1 + SMEM_OFFSET_BAND_BIAS_1;

    uint32_t smem_bar_qkv_tma_1 = smem_bar_qkv_1;
    uint32_t smem_bar_qkv_mma_1 = smem_bar_qkv_1 + 1 * 8 * (BUFFERS_Q_1 + BUFFERS_K_1 + BUFFERS_V_1 + BUFFERS_D_1 + 0);
    uint32_t smem_tile_id_1     = smem_bar_qkv_1 + 2 * 8 * (BUFFERS_Q_1 + BUFFERS_K_1 + BUFFERS_V_1 + BUFFERS_D_1 + 0);
    uint32_t smem_bar_tile_id_tma_1 = smem_tile_id_1 + 16;
    uint32_t smem_bar_bias_tma_1 = smem_bar_tile_id_tma_1 + 8;

    uint32_t smem_bar_bias_mma_1 = smem_bar_bias_tma_1 + 8 * 2;

    if (elect_one) {
        for (int i = wid; i < (BUFFERS_Q_1 + BUFFERS_K_1 + BUFFERS_V_1 + BUFFERS_D_1 + 0); i+=12) {
            smem_bar_init(smem_bar_qkv_tma_1 + i * 8, 1);
            smem_bar_init(smem_bar_qkv_mma_1 + i * 8, 2);
        }
    }
    __syncthreads();

    // mma_pipeline_op 1 code
    const int actual_num_tiles_1 = num_tiles_1;
    if (gid == 0) {
        reg_dealloc<40>();
        const uint32_t local_wid = wid % 4;

        uint32_t tile_id;
        uint32_t batch_coord_1 = 0, head_coord_1 = 0, q_row_coord = 0, rows_temp;
        uint32_t head_coord_from_grid_1, head_coord_k_1, head_coord_v_1, head_coord_q_offset_1;
        int actual_seqlen_kv_1 = attnDesc.s_kv, actual_seqlen_q_1 = attnDesc.s_q, oob_for_stats_1 = attnDesc.s_q;

        if (local_wid == 0) {
            uint32_t smem_bar_q_tma = smem_bar_qkv_tma_1;
            uint32_t smem_bar_k_tma = smem_bar_q_tma + 8 * BUFFERS_Q_1;
            uint32_t smem_bar_v_tma = smem_bar_k_tma + 8 * BUFFERS_K_1;

            uint32_t smem_bar_q_mma = smem_bar_qkv_mma_1;
            uint32_t smem_bar_k_mma = smem_bar_q_mma + 8 * BUFFERS_Q_1;
            uint32_t smem_bar_v_mma = smem_bar_k_mma + 8 * BUFFERS_K_1;

            uint32_t buffer_id_q  = 0;
            uint32_t buffer_id_kv = 0;
            uint32_t cnt_q        = 0;
            uint32_t cnt_kv       = 0;

            uint32_t sts_page_buffer_id = 0;
            uint32_t lds_page_buffer_id = 0;

            int cumulative_tiles = 0;
            if (tiw == 0) {
                tile_id = atomicAdd(tile_id_counter, 1);
            }
            tile_id = __shfl_sync(0xffffffff, tile_id, 0);

            named_barrier_wait(TILE_ID_SYNC_2_BARRIER_1, THREADS_ON_TILE_BARRIER_1);
            if (tiw == 0) {
                sts_32(smem_tile_id_1, &tile_id);
            }
            named_barrier_arrive(TILE_ID_SYNC_3_BARRIER_1, THREADS_ON_TILE_BARRIER_1);

            // Persistent loop over output tiles
#pragma unroll 1
            while (tile_id < actual_num_tiles_1) {
                fastDivMod(tiles_hr_div_1, tile_id, batch_coord_1, rows_temp);
                fastDivMod(tiles_r_div_1, rows_temp, head_coord_from_grid_1, q_row_coord);
                q_row_coord *= TILE_M_1 * 2;
                head_coord_1   = head_coord_from_grid_1;
                head_coord_k_1 = head_coord_1 / attnDesc.q_heads_per_k;
                head_coord_v_1 = head_coord_1 / attnDesc.q_heads_per_v;

                const int left_bound = 0;
                const int shift_right_bound = 0;

                int2 kv_loop_bounds = compute_kv_loop_bounds(q_row_coord, TILE_M_1 * 2, TILE_N_1, actual_seqlen_kv_1, actual_seqlen_q_1, shift_right_bound, left_bound);
                int kv_loop_left_bound = kv_loop_bounds.x;
                int kv_loop_right_bound = kv_loop_bounds.y;

                //global_load_shared_store_op 4 decls

                //global_load_shared_store_op 7 decls

                //global_load_shared_store_op 16 decls

                uint32_t local_smem_q         = smem_Q  + buffer_id_q  * SMEM_Q_1;
                uint32_t local_smem_bar_tma_q = smem_bar_q_tma + buffer_id_q  * 8;
                uint32_t local_smem_bar_mma_q = smem_bar_q_mma + buffer_id_q  * 8;

                if (kv_loop_left_bound < kv_loop_right_bound) {
                    uint32_t smem_bar_phase_q = cnt_q < BUFFERS_Q_1 ? 1 : 0;

                    cnt_q       = cnt_q       < (2 * BUFFERS_Q_1 - 1) ? (cnt_q       + 1) : 0;
                    buffer_id_q = buffer_id_q < (    BUFFERS_Q_1 - 1) ? (buffer_id_q + 1) : 0;

                    if (!smem_bar_peek(local_smem_bar_mma_q, smem_bar_phase_q)) {
                        smem_bar_wait(local_smem_bar_mma_q, smem_bar_phase_q);
                    }

                    smem_bar_set_transaction_count(local_smem_bar_tma_q, SMEM_Q_1, elect_one);
                    {
                        const uint32_t row_coord    = q_row_coord;

                        //global_load_shared_store_op 4 code
#pragma unroll
                        for (int i = 0; i < TILE_K_1 * BYTES_PER_ELEMENT_1; i += 128) {
                            utmaldg_4d_tiled(&tma_Q,
                            local_smem_q + i * TILE_M_1 * 2,
                            local_smem_bar_tma_q,
                            i / BYTES_PER_ELEMENT_1,
                            row_coord,
                            head_coord_1 + 0,
                            batch_coord_1,
                            elect_one);
                        }
                    }
                }

#pragma unroll 1
                for (int kv_loop = kv_loop_left_bound; kv_loop < kv_loop_right_bound; kv_loop++) {
                    const int p_col_coord = kv_loop * TILE_N_1;

                    uint32_t local_smem_k         = smem_K  + buffer_id_kv  * SMEM_K_1;
                    uint32_t local_smem_bar_tma_k = smem_bar_k_tma + buffer_id_kv  * 8;
                    uint32_t local_smem_bar_mma_k = smem_bar_k_mma + buffer_id_kv  * 8;

                    uint32_t local_smem_v         = smem_V  + buffer_id_kv  * SMEM_V_1;
                    uint32_t local_smem_bar_tma_v = smem_bar_v_tma + buffer_id_kv  * 8;

                    uint32_t local_smem_bar_mma_v = smem_bar_v_mma + buffer_id_kv  * 8;

                    uint32_t smem_bar_phase_kv = cnt_kv < BUFFERS_K_1 ? 1 : 0;

                    cnt_kv       = cnt_kv       < (2 * BUFFERS_K_1 - 1) ? (cnt_kv       + 1) : 0;
                    buffer_id_kv = buffer_id_kv < (    BUFFERS_K_1 - 1) ? (buffer_id_kv + 1) : 0;

                    uint32_t kv_col_base = (kv_loop + 1) * TILE_N_1;

                    if (!smem_bar_peek(local_smem_bar_mma_k, smem_bar_phase_kv)) {
                        smem_bar_wait(local_smem_bar_mma_k, smem_bar_phase_kv);
                    }
                    smem_bar_set_transaction_count(local_smem_bar_tma_k, SMEM_K_1, elect_one);

                    //global_load_shared_store_op 7 code
#pragma unroll
                    for (int i = 0; i < TILE_K_1 * BYTES_PER_ELEMENT_1; i += 128) {
                        utmaldg_4d_tiled(&tma_K,
                        local_smem_k + i * TILE_N_1,
                        local_smem_bar_tma_k,
                        i / BYTES_PER_ELEMENT_1,
                        p_col_coord,
                        head_coord_k_1,
                        batch_coord_1,
                        elect_one);
                    }
                    if (!smem_bar_peek(local_smem_bar_mma_v, smem_bar_phase_kv)) {
                        smem_bar_wait(local_smem_bar_mma_v, smem_bar_phase_kv);
                    }
                    smem_bar_set_transaction_count(local_smem_bar_tma_v, SMEM_V_1, elect_one);

                    //global_load_shared_store_op 16 code
#pragma unroll
                    for (int i = 0; i < TILE_K_1 * BYTES_PER_ELEMENT_1; i += 128 * CTA_MMA_1) {
                        utmaldg_4d_tiled(&tma_V,
                        local_smem_v + i * TILE_N_1,
                        local_smem_bar_tma_v,
                        0 + i / BYTES_PER_ELEMENT_1,
                        p_col_coord,
                        head_coord_v_1,
                        batch_coord_1,
                        elect_one);
                    }
                    lds_page_buffer_id ^= 1;
                }
                if (kv_loop_left_bound < kv_loop_right_bound) {
                    lds_page_buffer_id ^= 1;
                }

                if (tiw == 0) {
                    tile_id = atomicAdd(tile_id_counter, 1);
                }
                tile_id = __shfl_sync(0xffffffff, tile_id, 0);

                named_barrier_wait(TILE_ID_SYNC_2_BARRIER_1, THREADS_ON_TILE_BARRIER_1);
                if (tiw == 0) {
                    sts_32(smem_tile_id_1, &tile_id);
                }
                named_barrier_arrive(TILE_ID_SYNC_3_BARRIER_1, THREADS_ON_TILE_BARRIER_1);
            }
            named_barrier_wait(TILE_ID_SYNC_2_BARRIER_1, THREADS_ON_TILE_BARRIER_1);
        }
        reg_dealloc<40>();
    } else {
        reg_alloc<232>();
        named_barrier_arrive(TILE_ID_SYNC_2_BARRIER_1, THREADS_ON_TILE_BARRIER_1);

        const uint32_t local_wid = wid % 4;
        const uint32_t local_gid = gid % 2;

        uint32_t tile_id;
        uint32_t batch_coord_1, head_coord_1, q_row_coord, rows_temp;
        uint32_t head_coord_from_grid_1, head_coord_k_1, head_coord_v_1, head_coord_q_offset_1;
        int actual_seqlen_kv_1 = attnDesc.s_kv, actual_seqlen_q_1 = attnDesc.s_q, oob_for_stats_1 = attnDesc.s_q;

        uint32_t smem_bar_q_tma = smem_bar_qkv_tma_1;
        uint32_t smem_bar_k_tma = smem_bar_q_tma + 8 * BUFFERS_Q_1;
        uint32_t smem_bar_v_tma = smem_bar_k_tma + 8 * BUFFERS_K_1;

        uint32_t smem_bar_q_mma = smem_bar_qkv_mma_1;
        uint32_t smem_bar_k_mma = smem_bar_q_mma + 8 * BUFFERS_Q_1;
        uint32_t smem_bar_v_mma = smem_bar_k_mma + 8 * BUFFERS_K_1;

        uint32_t buffer_id_q  = 0;
        uint32_t buffer_id_kv = 0;
        uint32_t cnt_q        = 0;
        uint32_t cnt_kv       = 0;

        static constexpr float inv_ln2 = 1.4426950408889634074f;
        static constexpr float ln2     = 0.6931471805599453094f;

        Gmma_descriptor gmma_desc_q(create_gmma_desc_q_1());
        Gmma_descriptor gmma_desc_k(create_gmma_desc_k_1());
        Gmma_descriptor gmma_desc_v(create_gmma_desc_v_1());

        gmma_desc_q.set_smem<SMEM_Q_1, BUFFERS_Q_1>(smem_Q + local_gid * TILE_M_1 * 128);
        gmma_desc_k.set_smem<SMEM_K_1, BUFFERS_K_1>(smem_K);
        gmma_desc_v.set_smem<SMEM_V_1, BUFFERS_V_1>(smem_V);

        named_barrier_wait(TILE_ID_SYNC_3_BARRIER_1, THREADS_ON_TILE_BARRIER_1);
        if (tiw == 0) {
            lds_32(&tile_id, smem_tile_id_1);
        }
        named_barrier_arrive(TILE_ID_SYNC_2_BARRIER_1, THREADS_ON_TILE_BARRIER_1);
        tile_id = __shfl_sync(0xffffffff, tile_id, 0);

        // Persistent loop over output tiles
#pragma unroll 1
        while (tile_id < actual_num_tiles_1) {
            fastDivMod(tiles_hr_div_1, tile_id, batch_coord_1, rows_temp);
            fastDivMod(tiles_r_div_1, rows_temp, head_coord_from_grid_1, q_row_coord);
            q_row_coord = q_row_coord * TILE_M_1 * 2;
            head_coord_1   = head_coord_from_grid_1;
            head_coord_k_1 = head_coord_1 / attnDesc.q_heads_per_k;
            head_coord_v_1 = head_coord_1 / attnDesc.q_heads_per_v;
            const int left_bound = 0;
            const int shift_right_bound = 0;

            int2 kv_loop_bounds = compute_kv_loop_bounds(q_row_coord, TILE_M_1 * 2, TILE_N_1, actual_seqlen_kv_1, actual_seqlen_q_1, shift_right_bound, left_bound);
            int kv_loop_left_bound = kv_loop_bounds.x;
            int kv_loop_right_bound = kv_loop_bounds.y;
            q_row_coord += local_gid * TILE_M_1;

            const int causal_mask_col = (tiw % 4) * 2;
            const int causal_mask_row = q_row_coord + local_wid * 16 + (tiw / 4);

            // mma_loop_op 2 decls

            //global_load_shared_store_op 4 decls

            // shared_load_op 5 decls

            // shared_load_op 6 decls

            //global_load_shared_store_op 7 decls

            //mma_op 8 decls
            r32 reg_8_0[REGS_M_8][REGS_N_8][WARP_REGS_8];

            //pointwise_calc_op 9 decls

            //global_load_op 10 decls
            r32 reg_10_0[1];
            reg_10_0[0] = reinterpret_cast<const r32 &>(attn_scale);
            const int oob_M_10 = 1;
            const int oob_N_10 = 1;

            //mha_mask_op 11 decls

            //softmax_op 12 decls
            r32 reg_12_0[REGS_M_12][REGS_N_12][4];  // output for second gemm
            r32 reg_12_1[ROWS_PER_THREAD_12];  // acc_max
            r32 reg_12_2[ROWS_PER_THREAD_12];  // p_max
            r32 reg_12_3[ROWS_PER_THREAD_12];  // acc_sum
            r32 reg_12_4[ROWS_PER_THREAD_12];  // p_sum
            r32 reg_12_5[ROWS_PER_THREAD_12];  // softmax stats
            float beta_12[ROWS_PER_THREAD_12];

#pragma unroll
            for (int i = 0; i < ROWS_PER_THREAD_12; ++i) {
                reinterpret_cast<float &>(reg_12_1[i]) = NEG_INFINITY;
                reinterpret_cast<float &>(reg_12_3[i]) = 0.0f;
            }

            //global_store_op 13 decls
            int oob_M_13 = desc_max.dims[2];
            int row_13   = q_row_coord + local_wid * WARP_TILE_M_1 + (tiw / 4);
            char *ptr_13 = reinterpret_cast<char *>(d_max)
            + batch_coord_1 * desc_max.strides[0] * BYTES_PER_ELEMENT_13
            + head_coord_1 * desc_max.strides[1] * BYTES_PER_ELEMENT_13
            + row_13 * desc_max.strides[2] * BYTES_PER_ELEMENT_13;

            //global_store_op 14 decls
            int oob_M_14 = desc_sum_exp.dims[2];
            int row_14   = q_row_coord + local_wid * WARP_TILE_M_1 + (tiw / 4);
            char *ptr_14 = reinterpret_cast<char *>(d_sum_exp)
            + batch_coord_1 * desc_sum_exp.strides[0] * BYTES_PER_ELEMENT_14
)KERNEL"
    R"KERNEL(
            + head_coord_1 * desc_sum_exp.strides[1] * BYTES_PER_ELEMENT_14
            + row_14 * desc_sum_exp.strides[2] * BYTES_PER_ELEMENT_14;

            // shared_load_op 15 decls

            //global_load_shared_store_op 16 decls

            //mma_op 17 decls

            r32 reg_17_0[REGS_M_17][REGS_N_17][WARP_REGS_17];

            //global_store_op 20 decls
            static constexpr int cols_per_step_20 = (TILE_K_1 <= 128) ? TILE_K_1 : 64;
            uint32_t smem_20 = smem_O + (local_gid * 4 + local_wid) * 16 * cols_per_step_20 * BYTES_PER_ELEMENT_1;

            const int swizzled_row_20 = (tiw % 16);
            const int swizzled_col_20 = (tiw / 16);
            constexpr int stsm_per_tile_20 = 128 / (BYTES_PER_BANK_1 * 2);
            uint32_t stsm_base_20[stsm_per_tile_20];

#pragma unroll
            for (int n = 0; n < stsm_per_tile_20; ++n) {
                stsm_base_20[n] = smem_20 + ((swizzled_row_20 % 8) ^ (n * 2 + swizzled_col_20)) * BYTES_PER_BANK_1 + swizzled_row_20 * 128;
            }

            const int row_20 = q_row_coord + (local_wid % 4) * 16;

            uint32_t local_smem_bar_tma_q = smem_bar_q_tma + buffer_id_q  * 8;
            uint32_t local_smem_bar_mma_q = smem_bar_q_mma + buffer_id_q  * 8;

            if (local_gid == 1) {
                named_barrier_arrive(SOFTMAX_2_BARRIER_1, 256);
            }

            if (kv_loop_left_bound < kv_loop_right_bound) {
                uint32_t smem_bar_phase_q = cnt_q < BUFFERS_Q_1 ? 0 : 1;

                cnt_q       = cnt_q       < (2 * BUFFERS_Q_1 - 1) ? (cnt_q       + 1) : 0;
                buffer_id_q = buffer_id_q < (    BUFFERS_Q_1 - 1) ? (buffer_id_q + 1) : 0;

                if (!smem_bar_peek(local_smem_bar_tma_q, smem_bar_phase_q)) {
                    smem_bar_wait(local_smem_bar_tma_q, smem_bar_phase_q);
                }
            }

            r32 reg_1_0[REGS_M_1][REGS_O_1][WARP_REGS_1];
            memset(&reg_1_0[0][0][0], 0, sizeof(reg_1_0));

            named_barrier_wait(MATH_WORKGROUP_1, 256);
#pragma unroll 1
            for (int kv_loop = kv_loop_left_bound; kv_loop < kv_loop_right_bound; kv_loop++) {
                static constexpr int kMulPipeCount      = REGS_M_1 * REGS_N_1 * WARP_REGS_1;

                static constexpr int kSubtractPipeCount = 6;
                static constexpr int kConvertPipeCount  = 4;
                static constexpr int oBlockSize         = 2;

                const int p_col_coord = kv_loop * TILE_N_1;
                const int causal_mask_col = p_col_coord + (tiw % 4) * 2;
                const int causal_mask_row = q_row_coord + local_wid * 16 + (tiw / 4);

                uint32_t local_smem_bar_tma_k = smem_bar_k_tma + buffer_id_kv  * 8;
                uint32_t local_smem_bar_mma_k = smem_bar_k_mma + buffer_id_kv  * 8;

                uint32_t local_smem_bar_tma_v = smem_bar_v_tma + buffer_id_kv  * 8;
                uint32_t local_smem_bar_mma_v = smem_bar_v_mma + buffer_id_kv  * 8;

                uint32_t smem_bar_phase_kv = cnt_kv < BUFFERS_K_1 ? 0 : 1;

                cnt_kv       = cnt_kv       < (2 * BUFFERS_K_1 - 1) ? (cnt_kv       + 1) : 0;
                buffer_id_kv = buffer_id_kv < (    BUFFERS_K_1 - 1) ? (buffer_id_kv + 1) : 0;

                memset(&reg_8_0[0][0][0], 0, sizeof(reg_8_0));

                if (!smem_bar_peek(local_smem_bar_tma_k, smem_bar_phase_kv)) {
                    smem_bar_wait(local_smem_bar_tma_k, smem_bar_phase_kv);
                }

                warpgroup_arrive();
#pragma unroll
                for (int k = 0; k < REGS_K_1; ++k) {
                    if (k == REGS_K_1 - 1) {
                        BMM_S_GMMA_ISB(gmma_desc_q.desc, gmma_desc_k.desc, reinterpret_cast<uint32_t *>(reg_8_0[0]));
                    } else {
                        BMM_S_GMMA(gmma_desc_q.desc, gmma_desc_k.desc, reinterpret_cast<uint32_t *>(reg_8_0[0]));
                    }
                    int2 &tmp_desc_q = reinterpret_cast<int2 &>(gmma_desc_q.desc);
                    tmp_desc_q.x += BYTES_PER_GMMA_K_NO_4LSB_1;
                    if ((k % 4) == 3) {
                        tmp_desc_q.x -= BYTES_PER_GMMA_K_NO_4LSB_1 * 4;
                        tmp_desc_q.x += ((128*TILE_M_1*2) >> 4);
                    }
                    int2 &tmp_desc_k = reinterpret_cast<int2 &>(gmma_desc_k.desc);
                    tmp_desc_k.x += BYTES_PER_GMMA_K_NO_4LSB_1;
                    if ((k % 4) == 3) {
                        tmp_desc_k.x -= BYTES_PER_GMMA_K_NO_4LSB_1 * 4;
                        tmp_desc_k.x += ((128*TILE_N_1) >> 4);
                    }
                }
                {
                    int2 &tmp_desc_q = reinterpret_cast<int2 &>(gmma_desc_q.desc);
                    tmp_desc_q.x -= ((128 * TILE_M_1 * 2) >> 4) * (REGS_K_1 / 4);
                    int2 &tmp_desc_k = reinterpret_cast<int2 &>(gmma_desc_k.desc);
                    tmp_desc_k.x -= ((128 * TILE_N_1) >> 4) * (REGS_K_1 / 4);
                }
                gmma_desc_k.increment_smem_buffer<SMEM_K_1, BUFFERS_K_1>();
                warpgroup_wait<0>();

                if (local_wid == 0 && elect_one) {
                    smem_bar_arrive(local_smem_bar_mma_k);
                }

                uint32_t wait_v_done = smem_bar_peek(local_smem_bar_tma_v, smem_bar_phase_kv);

                //mha_mask_op 11 code
                if (kv_loop >= kv_loop_right_bound - 1)
                {
#pragma unroll
                    for (int m = 0; m < REGS_M_11; ++m) {
#pragma unroll
                        for (int n = 0; n < REGS_N_11; ++n) {
#pragma unroll
                            for (int i = 0; i < WARP_REGS_11; ++i) {
                                int row =  causal_mask_row + m * WARPS_PER_GROUP_0 * WARP_TILE_M_11 + ((i / 2) % 2) * 8;
                                int col = causal_mask_col + n * WARP_TILE_N_11 + i % 2 + (i / 4) * 8;
                                bool mask = compute_diagonal_band_mask_11(row, col, actual_seqlen_kv_1, actual_seqlen_q_1, shift_right_bound, left_bound);
                                if (!mask) {
                                    reinterpret_cast<float &>(reg_8_0[m][n][i]) = neg_infinity;
                                }
                            }
                        }
                    }
                }

                //global_load_op 10 code

                //mha_mask_op 11 code
                if (kv_loop >= kv_loop_right_bound - 1)
                {
#pragma unroll
                    for (int m = 0; m < REGS_M_11; ++m) {
#pragma unroll
                        for (int n = 0; n < REGS_N_11; ++n) {
#pragma unroll
                            for (int i = 0; i < WARP_REGS_11; ++i) {
                                int row =  causal_mask_row + m * WARPS_PER_GROUP_0 * WARP_TILE_M_11 + ((i / 2) % 2) * 8;
                                int col = causal_mask_col + n * WARP_TILE_N_11 + i % 2 + (i / 4) * 8;
                                bool mask = compute_diagonal_band_mask_11(row, col, actual_seqlen_kv_1, actual_seqlen_q_1, shift_right_bound, left_bound);
                                if (!mask) {
                                    reinterpret_cast<float &>(reg_8_0[m][n][i]) = neg_infinity;
                                }
                            }
                        }
                    }
                }

                //softmax_op 12 code
                static constexpr float inv_ln2_12 = 1.4426950408889634074f;

                float scaled_inv_ln2_12 = inv_ln2_12 * reinterpret_cast<const float &>(reg_10_0[0]);

                r32 *flattened_s      = reinterpret_cast<r32 *>(reg_8_0);
                r32 *flattened_s_fp16 = reinterpret_cast<r32 *>(reg_12_0);
                r32 *flattened_o      = reinterpret_cast<r32 *>(reg_1_0);

                float tmp_max_level_0[REGS_N_12][4];
                float tmp_max_level_1[REGS_N_12][2];
                float tmp_max_level_2[REGS_N_12 / 2][2];
#pragma unroll
                for (int m = 0; m < REGS_M_12; ++m) {
#pragma unroll
                    for (int n = 0; n < REGS_N_12; ++n) {
#pragma unroll
                        for (int i = 0; i < WARP_REGS_12; i+=2) {
                            tmp_max_level_0[n][i/2] = fmaxf(reinterpret_cast<const float &>(reg_8_0[m][n][i+0]),
                            reinterpret_cast<const float &>(reg_8_0[m][n][i+1]));
                        }
                    }
                }
                // FMUL
#pragma unroll
                for (int i = 0; i < kMulPipeCount; i++) {
                    reinterpret_cast<float &>(flattened_s[i]) = __fmul_rn(scaled_inv_ln2_12, reinterpret_cast<float &>(flattened_s[i]));
                }
                {
#pragma unroll
                    for (int n = 0; n < REGS_N_12; ++n) {
#pragma unroll
                        for (int i = 0; i < 2; i++) {
                            tmp_max_level_1[n][i] = fmaxf(tmp_max_level_0[n][i + 0],
                            tmp_max_level_0[n][i + 2]);
                        }
                    }
                }
                {
#pragma unroll
                    for (int n = 0; n < REGS_N_12; n+=2) {
                        tmp_max_level_2[n/2][0] = fmaxf(tmp_max_level_1[n][0], tmp_max_level_1[n+1][0]);
                        tmp_max_level_2[n/2][1] = fmaxf(tmp_max_level_1[n][1], tmp_max_level_1[n+1][1]);
                    }
                }
                if (TILE_N_1 == 256) {
                    float tmp_max_level_3[REGS_N_12 / 4][2];
                    float tmp_max_level_4[FORT_MAX(1,REGS_N_12 / 8)][2];
#pragma unroll
                    for (int n = 0; n < REGS_N_12; n+=4) {
                        tmp_max_level_3[n/4][0] = fmaxf(tmp_max_level_2[n/2][0], tmp_max_level_2[n/2+1][0]);
                        tmp_max_level_3[n/4][1] = fmaxf(tmp_max_level_2[n/2][1], tmp_max_level_2[n/2+1][1]);
                    }
#pragma unroll
                    for (int n = 0; n < REGS_N_12; n+=8) {
                        tmp_max_level_4[n/8][0] = fmaxf(tmp_max_level_3[n/4][0], tmp_max_level_3[n/4+1][0]);
                        tmp_max_level_4[n/8][1] = fmaxf(tmp_max_level_3[n/4][1], tmp_max_level_3[n/4+1][1]);
                    }
                    reinterpret_cast<float &>(reg_12_2[0]) = fmaxf(tmp_max_level_4[0][0], tmp_max_level_4[1][0]);
                    reinterpret_cast<float &>(reg_12_2[1]) = fmaxf(tmp_max_level_4[0][1], tmp_max_level_4[1][1]);
                } else if (TILE_N_1 == 128) {
                    float tmp_max_level_3[REGS_N_12 / 4][2];
#pragma unroll
                    for (int n = 0; n < REGS_N_12; n+=4) {
                        tmp_max_level_3[n/4][0] = fmaxf(tmp_max_level_2[n/2][0], tmp_max_level_2[n/2+1][0]);
                        tmp_max_level_3[n/4][1] = fmaxf(tmp_max_level_2[n/2][1], tmp_max_level_2[n/2+1][1]);
                    }
                    reinterpret_cast<float &>(reg_12_2[0]) = fmaxf(tmp_max_level_3[0][0], tmp_max_level_3[1][0]);
                    reinterpret_cast<float &>(reg_12_2[1]) = fmaxf(tmp_max_level_3[0][1], tmp_max_level_3[1][1]);
                } else {
                    reinterpret_cast<float &>(reg_12_2[0]) = fmaxf(tmp_max_level_2[0][0], tmp_max_level_2[1][0]);
                    reinterpret_cast<float &>(reg_12_2[1]) = fmaxf(tmp_max_level_2[0][1], tmp_max_level_2[1][1]);
                }

                // Apply the functor for each row inside each group of 4 threads.
#pragma unroll
                for (int m = 0; m < ROWS_PER_THREAD_12; ++m) {
                    r32 tmp_0 = __shfl_xor_sync(uint32_t(-1), reg_12_2[m], 1);

                    reinterpret_cast<float &>(reg_12_2[m]) = fmaxf(reinterpret_cast<const float &>(reg_12_2[m]),
                    reinterpret_cast<const float &>(tmp_0));

                    tmp_0 = __shfl_xor_sync(uint32_t(-1), reg_12_2[m], 2);

                    reinterpret_cast<float &>(reg_12_2[m]) = fmaxf(reinterpret_cast<const float &>(reg_12_2[m]),
                    reinterpret_cast<const float &>(tmp_0)) * scaled_inv_ln2_12;
                }

                // Update acc_max scale of flash attention
#pragma unroll
                for (int m = 0; m < ROWS_PER_THREAD_12; ++m) {
                    r32 curr_max = reg_12_1[m];
                    reinterpret_cast<float &>(reg_12_1[m]) = fmaxf(reinterpret_cast<const float &>(reg_12_2[m]),
                    reinterpret_cast<const float &>(curr_max));
                    reg_12_2[m] = curr_max;
                }

                float alpha_12[ROWS_PER_THREAD_12];
#pragma unroll
                for (int m = 0; m < ROWS_PER_THREAD_12; ++m) {
                    float p_max   = reinterpret_cast<const float &>(reg_12_2[m]);
                    float acc_max = reinterpret_cast<const float &>(reg_12_1[m]);

                    // disable FMA for scale * P - max
                    float tmp                            = (acc_max == NEG_INFINITY) ? p_max : __fsub_rn(p_max, acc_max);
                    reinterpret_cast<r32 &>(alpha_12[m]) = fp32_exp2(reinterpret_cast<r32 &>(tmp));
                }

#pragma unroll
                for (int i = 0; i < ROWS_PER_THREAD_12; ++i) {
                    reinterpret_cast<float &>(reg_12_4[i]) = 0.0f;
                }

                // Broadcast scaled_max and alpha
                float flattened_scaled_max[REGS_M_12 * REGS_N_12 * WARP_REGS_12];
                float flattened_alpha[REGS_M_12 * REGS_O_1 * WARP_REGS_12];
#pragma unroll
                for (int m = 0; m < REGS_M_12; ++m) {
#pragma unroll
                    for (int n = 0; n < REGS_N_12; ++n) {
#pragma unroll
                        for (int i = 0; i < WARP_REGS_12; ++i) {
                            int row_indx =  m * 2 + (i / 2) % 2;
                            float max = reinterpret_cast<const float &>(reg_12_1[row_indx]);
                            flattened_scaled_max[m * REGS_N_12 * WARP_REGS_12 + n * WARP_REGS_12 + i] = (max == NEG_INFINITY ? 0.0f : max);
                        }
                    }
                }

#pragma unroll
                for (int m = 0; m < REGS_M_12; ++m) {
#pragma unroll
                    for (int n = 0; n < REGS_O_1; ++n) {
#pragma unroll
                        for (int i = 0; i < WARP_REGS_12; ++i) {
                            int row_indx =  m * 2 + (i / 2) % 2;
                            flattened_alpha[m * REGS_O_1 * WARP_REGS_12 + n * WARP_REGS_12 + i] = alpha_12[row_indx];
                        }
                    }
                }

                cfence();
                if (local_gid == 1) {
                    named_barrier_wait(SOFTMAX_1_BARRIER_1, 256);
                } else {
                    named_barrier_wait(SOFTMAX_2_BARRIER_1, 256);
                }
                cfence();

                // FSUB
#pragma unroll
                for (int i = 0; i < kSubtractPipeCount; i++) {
)KERNEL"
    R"KERNEL(
                    reinterpret_cast<float &>(flattened_s[i]) = __fsub_rn(reinterpret_cast<float &>(flattened_s[i]), flattened_scaled_max[i]);
                }

#pragma unroll

                for (int i = 0; i < REGS_M_12 * REGS_N_12 * WARP_REGS_12; i+=2) {
                    int m_0  = (i-kConvertPipeCount+0) / (REGS_N_12 * WARP_REGS_12);
                    int ii_0 = (i-kConvertPipeCount+0) % WARP_REGS_12;
                    int m_1  = (i-kConvertPipeCount+1) / (REGS_N_12 * WARP_REGS_12);
                    int ii_1 = (i-kConvertPipeCount+1) % WARP_REGS_12;
                    int row_indx_0 =  m_0 * 2 + (ii_0 / 2) % 2;
                    int row_indx_1 =  m_1 * 2 + (ii_1 / 2) % 2;
                    // cast and FADD
                    if (i >= kConvertPipeCount) {
                        fp32x2_to_bf16x2(&flattened_s_fp16[(i - kConvertPipeCount)/2], &flattened_s[i - kConvertPipeCount]);
                        reinterpret_cast<float &>(reg_12_4[row_indx_0]) += reinterpret_cast<float &>(flattened_s[i-kConvertPipeCount+0]);
                        reinterpret_cast<float &>(reg_12_4[row_indx_1]) += reinterpret_cast<float &>(flattened_s[i-kConvertPipeCount+1]);
                    }
                    cfence();

                    // exp
                    flattened_s[i+0] = fp32_exp2(flattened_s[i+0]);

                    // O update
#pragma unroll
                    for (int j = 0; j < oBlockSize; j++) {
                        if (oBlockSize*i+j < REGS_O_1 * REGS_M_1 * WARP_REGS_1) {
                            reinterpret_cast<float &>(flattened_o[oBlockSize*i+j]) *= flattened_alpha[oBlockSize*i+j];
                        }
                    }

                    // fmul
                    if (i + kMulPipeCount < REGS_M_12 * REGS_N_12 * WARP_REGS_12) {
                        reinterpret_cast<float &>(flattened_s[i+kMulPipeCount+0]) = __fmul_rn(scaled_inv_ln2_12, reinterpret_cast<float &>(flattened_s[i+kMulPipeCount+0]));
                        reinterpret_cast<float &>(flattened_s[i+kMulPipeCount+1]) = __fmul_rn(scaled_inv_ln2_12, reinterpret_cast<float &>(flattened_s[i+kMulPipeCount+1]));
                    }
                    // fsub

                    if (i + kSubtractPipeCount < REGS_M_12 * REGS_N_12 * WARP_REGS_12) {
                        reinterpret_cast<float &>(flattened_s[i+kSubtractPipeCount+0]) = __fsub_rn(reinterpret_cast<float &>(flattened_s[i+kSubtractPipeCount+0]), flattened_scaled_max[i+kSubtractPipeCount+0]);
                        reinterpret_cast<float &>(flattened_s[i+kSubtractPipeCount+1]) = __fsub_rn(reinterpret_cast<float &>(flattened_s[i+kSubtractPipeCount+1]), flattened_scaled_max[i+kSubtractPipeCount+1]);
                    }

                    // exp
                    flattened_s[i+1] = fp32_exp2(flattened_s[i+1]);

                    // O update
#pragma unroll
                    for (int j = 0; j < oBlockSize; j++) {
                        if (oBlockSize+oBlockSize*i+j < REGS_O_1 * REGS_M_1 * WARP_REGS_1) {
                            reinterpret_cast<float &>(flattened_o[oBlockSize+oBlockSize*i+j]) *= flattened_alpha[oBlockSize+oBlockSize*i+j];
                        }
                    }
                }
                if (local_gid == 0) {
                    named_barrier_arrive(SOFTMAX_1_BARRIER_1, 256);
                } else {
                    named_barrier_arrive(SOFTMAX_2_BARRIER_1, 256);
                }
                cfence();

#pragma unroll
                for (int i = REGS_M_12 * REGS_N_12 * WARP_REGS_12 - kConvertPipeCount; i < REGS_M_12 * REGS_N_12 * WARP_REGS_12; i+=2) {
                    int m_0  = (i + 0) / (REGS_N_12 * WARP_REGS_12);
                    int ii_0 = (i + 0) % WARP_REGS_12;
                    int m_1  = (i + 1) / (REGS_N_12 * WARP_REGS_12);
                    int ii_1 = (i + 1) % WARP_REGS_12;
                    int row_indx_0 =  m_0 * 2 + (ii_0 / 2) % 2;
                    int row_indx_1 =  m_1 * 2 + (ii_1 / 2) % 2;
                    fp32x2_to_bf16x2(&flattened_s_fp16[i/2], &flattened_s[i]);
                    reinterpret_cast<float &>(reg_12_4[row_indx_0]) += reinterpret_cast<float &>(flattened_s[i+0]);
                    reinterpret_cast<float &>(reg_12_4[row_indx_1]) += reinterpret_cast<float &>(flattened_s[i+1]);
                }

                if (!wait_v_done) {
                    smem_bar_wait(local_smem_bar_tma_v, smem_bar_phase_kv);
                }

                warpgroup_arrive();
#pragma unroll
                for (int k = 0; k < TILE_N_1 / GMMA_TILE_K_1; ++k) {
                    if (k == (TILE_N_1 / GMMA_TILE_K_1) - 1) {
                        BMM_O_GMMA_ISB(reg_12_0[0][k], gmma_desc_v.desc, reinterpret_cast<uint32_t *>(reg_1_0[0]));
                    } else {
                        BMM_O_GMMA(reg_12_0[0][k], gmma_desc_v.desc, reinterpret_cast<uint32_t *>(reg_1_0[0]));
                    }
                    int2 &tmp_desc_v = reinterpret_cast<int2 &>(gmma_desc_v.desc);
                    tmp_desc_v.x += BYTES_PER_GMMA_K_NO_4LSB_TRANS_1;
                }
                {
                    int2 &tmp_desc_v = reinterpret_cast<int2 &>(gmma_desc_v.desc);
                    tmp_desc_v.x -= BYTES_PER_GMMA_K_NO_4LSB_TRANS_1 * (TILE_N_1 / GMMA_TILE_K_1);
                }
                gmma_desc_v.increment_smem_buffer<SMEM_V_1, BUFFERS_V_1>();

                warpgroup_wait<0>();
                if (local_wid == 0 && elect_one) {
                    smem_bar_arrive(local_smem_bar_mma_v);
                }

                // Update acc_sum of flash attention
#pragma unroll
                for (int m = 0; m < ROWS_PER_THREAD_12; ++m) {
                    float p_sum   = reinterpret_cast<const float &>(reg_12_4[m]);
                    float acc_sum = reinterpret_cast<const float &>(reg_12_3[m]);

                    reinterpret_cast<float &>(reg_12_3[m]) = reinterpret_cast<float &>(alpha_12[m]) * acc_sum + p_sum;
                }
            }
            if (kv_loop_left_bound < kv_loop_right_bound) {
                gmma_desc_q.increment_smem_buffer<SMEM_Q_1, BUFFERS_Q_1>();

                if (local_wid == 0 && elect_one) {
                    smem_bar_arrive(local_smem_bar_mma_q);
                }
            }
            if (local_gid == 0) {
                named_barrier_arrive(SOFTMAX_2_BARRIER_1, 256);
            }
#pragma unroll
            for (int m = 0; m < ROWS_PER_THREAD_12; ++m) {
                r32 tmp_0 = __shfl_xor_sync(uint32_t(-1), reg_12_3[m], 1);

                reinterpret_cast<float &>(reg_12_3[m]) = reinterpret_cast<const float &>(reg_12_3[m]) +
                reinterpret_cast<const float &>(tmp_0);

                tmp_0 = __shfl_xor_sync(uint32_t(-1), reg_12_3[m], 2);

                reinterpret_cast<float &>(reg_12_3[m]) = reinterpret_cast<const float &>(reg_12_3[m]) +
                reinterpret_cast<const float &>(tmp_0);

                float sum = reinterpret_cast<float &>(reg_12_3[m]);
                reinterpret_cast<float &>(reg_12_1[m]) *= ln2;

                reinterpret_cast<float &>(reg_12_5[m]) = (sum == 0.f) ? NEG_INFINITY : reinterpret_cast<const float &>(reg_12_1[m]) + __logf(sum);

                beta_12[m] = (sum == 0.f) ? 0.f : 1.f / sum;
            }

#pragma unroll
            for (int m = 0; m < REGS_M_1; ++m) {
#pragma unroll
                for (int n = 0; n < REGS_O_1; ++n) {
#pragma unroll
                    for (int i = 0; i < WARP_REGS_1; ++i) {
                        int row_indx =  m * 2 + (i / 2) % 2;

                        reinterpret_cast<float &>(reg_1_0[m][n][i]) *= beta_12[row_indx];
                    }
                }
            }

            //global_store_op 13 code
#pragma unroll
            for (int m = 0; m < ROWS_PER_THREAD_13; ++m) {
                const int row_offset = (m % 2) * THREADS_PER_WARP_TILE_M_13 + (m / 2) * WARPS_M_13 * WARP_TILE_M_13;

                if (row_13 + row_offset < oob_M_13 && (tiw % 4) == 0) {
                    stg_32(ptr_13 + row_offset * desc_max.strides[2] * BYTES_PER_ELEMENT_13, &reg_12_1[m]);
                }
            }

            //global_store_op 14 code
#pragma unroll
            for (int m = 0; m < ROWS_PER_THREAD_14; ++m) {
                const int row_offset = (m % 2) * THREADS_PER_WARP_TILE_M_14 + (m / 2) * WARPS_M_14 * WARP_TILE_M_14;

                if (row_14 + row_offset < oob_M_14 && (tiw % 4) == 0) {
                    stg_32(ptr_14 + row_offset * desc_sum_exp.strides[2] * BYTES_PER_ELEMENT_14, &reg_12_3[m]);
                }
            }

            tmastg_wait();

            //global_store_op 20 code
#pragma unroll
            for (int s = 0; s < TILE_O_1 / 64; ++s) {
#pragma unroll
                for (int n = 0; n < stsm_per_tile_20; ++n) {
                    uint32_t tmp[4];
                    fp32x2_to_bf16x2(&tmp[0], &reg_1_0[0][s * stsm_per_tile_20 + n][0]);
                    fp32x2_to_bf16x2(&tmp[1], &reg_1_0[0][s * stsm_per_tile_20 + n][2]);
                    fp32x2_to_bf16x2(&tmp[2], &reg_1_0[0][s * stsm_per_tile_20 + n][4]);
                    fp32x2_to_bf16x2(&tmp[3], &reg_1_0[0][s * stsm_per_tile_20 + n][6]);
                    stsm_x4(stsm_base_20[n] + s * 64 * 16 * BYTES_PER_ELEMENT_1, tmp);
                }
            }
            fence_view_async_shared();

            int head_coord_loc = 0;
            int row = row_20;
            int head = head_coord_1;
#pragma unroll
            for (int s = 0; s < TILE_K_1 / 64; ++s) {
                utmastg_4d_tiled(&tma_O,
                smem_20 + s * 64 * 16 * BYTES_PER_ELEMENT_1,
                s * 64,
                row,
                head + head_coord_loc,
                batch_coord_1);
            }
            tmastg_arrive();

            named_barrier_wait(TILE_ID_SYNC_3_BARRIER_1, THREADS_ON_TILE_BARRIER_1);
            if (tiw == 0) {
                lds_32(&tile_id, smem_tile_id_1);
            }
            named_barrier_arrive(TILE_ID_SYNC_2_BARRIER_1, THREADS_ON_TILE_BARRIER_1);
            tile_id = __shfl_sync(0xffffffff, tile_id, 0);
        }
        tmastg_wait();
        reg_alloc<232>();
    }
}
)KERNEL";
inline constexpr size_t d64_fprop_source_len = sizeof(d64_fprop_source) - 1;

inline constexpr const char d64_fprop_flags[] = R"FLAGS(--gpu-architecture=sm_90a
--std=c++17
--define-macro=__x86_64__
-w
--define-macro=__CUDACC_RTC__
-default-device
--use_fast_math 
)FLAGS";
inline constexpr size_t d64_fprop_flags_len   = sizeof(d64_fprop_flags) - 1;

}  // namespace generated
}  // namespace experimental
}  // namespace cudnn_frontend

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
