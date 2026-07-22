// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/fmha.hpp"

#include "bias.hpp"
#include "mask.hpp"
#include "quant.hpp"
#include "rotary.hpp"

#include <type_traits>
#include <utility>
#include <variant>

struct FmhaFwdFp32
{
};

struct FmhaFwdFp16
{
};

struct FmhaFwdBf16
{
};

struct FmhaFwdFp8
{
};

struct FmhaFwdBf8
{
};

struct FmhaFwdFp8Fp16
{
};

struct FmhaFwdFp8Bf16
{
};

struct FmhaFwdFp8Fp32
{
};

struct FmhaFwdMxFp8
{
};

struct FmhaFwdMxFp4
{
};

template <typename DataType>
struct FmhaFwdTypeConfig;

template <>
struct FmhaFwdTypeConfig<FmhaFwdFp32>
{
    using QDataType             = float;
    using KDataType             = float;
    using VDataType             = float;
    using BiasDataType          = float;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = float; // data type for A matrix of second gemm
    using OaccDataType          = float; // data type for second gemm accumulation
    using ODataType             = float;
};

template <>
struct FmhaFwdTypeConfig<FmhaFwdFp16>
{
    using QDataType             = ck_tile::half_t;
    using KDataType             = ck_tile::half_t;
    using VDataType             = ck_tile::half_t;
    using BiasDataType          = ck_tile::half_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::half_t; // data type for A matrix of second gemm
    using OaccDataType          = float;           // data type for second gemm accumulation
    using ODataType             = ck_tile::half_t;
};

template <>
struct FmhaFwdTypeConfig<FmhaFwdBf16>
{
    using QDataType             = ck_tile::bf16_t;
    using KDataType             = ck_tile::bf16_t;
    using VDataType             = ck_tile::bf16_t;
    using BiasDataType          = ck_tile::bf16_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::bf16_t; // data type for A matrix of second gemm
    using OaccDataType          = float;           // data type for second gemm accumulation
    using ODataType             = ck_tile::bf16_t;
};

template <>
struct FmhaFwdTypeConfig<FmhaFwdFp8>
{
    using QDataType             = ck_tile::fp8_t;
    using KDataType             = ck_tile::fp8_t;
    using VDataType             = ck_tile::fp8_t;
    using BiasDataType          = float;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::fp8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;          // data type for second gemm accumulation
    using ODataType             = ck_tile::fp8_t;
};

template <>
struct FmhaFwdTypeConfig<FmhaFwdBf8>
{
    using QDataType             = ck_tile::bf8_t;
    using KDataType             = ck_tile::bf8_t;
    using VDataType             = ck_tile::bf8_t;
    using BiasDataType          = ck_tile::bf8_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::bf8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;          // data type for second gemm accumulation
    using ODataType             = ck_tile::bf8_t;
};

template <>
struct FmhaFwdTypeConfig<FmhaFwdFp8Bf16>
{
    using QDataType             = ck_tile::fp8_t;
    using KDataType             = ck_tile::fp8_t;
    using VDataType             = ck_tile::fp8_t;
    using BiasDataType          = float;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::fp8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;          // data type for second gemm accumulation
    using ODataType             = ck_tile::bf16_t;
};

template <>
struct FmhaFwdTypeConfig<FmhaFwdFp8Fp32>
{
    using QDataType             = ck_tile::fp8_t;
    using KDataType             = ck_tile::fp8_t;
    using VDataType             = ck_tile::fp8_t;
    using BiasDataType          = float;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::fp8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;          // data type for second gemm accumulation
    using ODataType             = float;
};

template <>
struct FmhaFwdTypeConfig<FmhaFwdMxFp8>
{
    using QDataType             = ck_tile::fp8_t;
    using KDataType             = ck_tile::fp8_t;
    using VDataType             = ck_tile::fp8_t;
    using BiasDataType          = float;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::fp8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;          // data type for second gemm accumulation
    using ODataType             = float;

    using QScaleDataType = ck_tile::e8m0_t;
    using KScaleDataType = ck_tile::e8m0_t;
    using VScaleDataType = ck_tile::e8m0_t;
    using PScaleDataType = ck_tile::e8m0_t;

    static constexpr ck_tile::index_t kQKScaleGranularity = 32;
    static constexpr ck_tile::index_t kVScaleGranularity  = 32;
};

template <>
struct FmhaFwdTypeConfig<FmhaFwdMxFp4>
{
    using QDataType             = ck_tile::pk_fp4_t;
    using KDataType             = ck_tile::pk_fp4_t;
    using VDataType             = ck_tile::pk_fp4_t;
    using BiasDataType          = float;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::pk_fp4_t; // data type for A matrix of second gemm
    using OaccDataType          = float;             // data type for second gemm accumulation
    using ODataType             = float;

    using QScaleDataType = ck_tile::e8m0_t;
    using KScaleDataType = ck_tile::e8m0_t;
    using VScaleDataType = ck_tile::e8m0_t;
    using PScaleDataType = ck_tile::e8m0_t;

    static constexpr ck_tile::index_t kQKScaleGranularity = 32;
    static constexpr ck_tile::index_t kVScaleGranularity  = 32;
};

struct FmhaMasks
{
    using NoMask      = ck_tile::GenericAttentionMask<false>;
    using GenericMask = ck_tile::GenericAttentionMask<true, true>;
    using CausalMask  = ck_tile::GenericAttentionMask<true, false>;
};

// runtime args, some will passed to karg, some will used to compute grids/blocks
struct fmha_fwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    const void* q_descale_ptr;
    const void* k_descale_ptr;
    const void* v_descale_ptr;
    void* rand_val_ptr;
    void* lse_ptr;
    void* o_ptr;

    // Usage notes for sequence length pointer parameters:
    //
    // [Note: Define "Group mode" vs "Batch mode" here if possible, e.g., "Group mode handles
    // MQA/GQA..."]
    //
    // With padding:
    //   Group mode:
    //     - seqstart_q_ptr, seqstart_k_ptr: Record cumulative physical (including padding) sequence
    //     lengths. [array size: batch + 1]
    //     - seqlen_q_ptr/seqlen_k_ptr: Records logical (excluding padding) length for each
    //     sequence. [array size: batch]
    //     - cu_seqlen_q_ptr/cu_seqlen_k_ptr: Records cumulative logical (excluding padding)
    //     sequence lengths. [array size: batch + 1]
    //     - seqlen_q_ptr (per-sequence) and cu_seqlen_q_ptr (cumulative logical) are mutually
    //     exclusive. Use one set, not both.
    //
    //   Batch mode:
    //     - cu_seqlen_q_ptr/cu_seqlen_k_ptr: Records cumulative logical (excluding padding)
    //     sequence lengths. [array size: batch + 1]
    //     - seqstart_* and seqlen_* pointers must be nullptr.
    //
    // Without padding:
    //   (Note: Physical length equals logical length)
    //
    //   Group mode:
    //     - seqstart_q_ptr, seqstart_k_ptr: Record cumulative physical sequence lengths. [array
    //     size: batch + 1]
    //     - seqlen_q_ptr/seqlen_k_ptr and cu_seqlen_q_ptr/cu_seqlen_k_ptr must be nullptr.
    //
    //   Batch mode:
    //     - All sequence length pointers (seqstart_*, seqlen_*, cu_seqlen_*) must be nullptr.
    //
    const void* seqstart_q_ptr =
        nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
    const void* seqstart_k_ptr =
        nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
    const void* seqlen_q_ptr = nullptr;    // Per-sequence logical (excluding padding) length array
                                           // [batch]. (Used in Group mode with padding)
    const void* seqlen_k_ptr = nullptr;    // Per-sequence logical (excluding padding) length array
                                           // [batch]. (Used in Group mode with padding)
    const void* cu_seqlen_q_ptr = nullptr; // Cumulative logical (excluding padding) sequence length
                                           // array [batch + 1]. (Used with padding)
    const void* cu_seqlen_k_ptr = nullptr; // Cumulative logical (excluding padding) sequence length
                                           // array [batch + 1]. (Used with padding)
    const void* block_scale_seqstart_q_ptr;
    const void* block_scale_seqstart_k_ptr;
    const void* seqstart_v_scale_ptr;
    const void* sink_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;
    ck_tile::index_t num_head_q_total = 0;
    ck_tile::index_t head_start       = 0;

    float scale_s;
    float logits_soft_cap;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_o;
    ck_tile::index_t stride_q_descale;
    ck_tile::index_t stride_k_descale;
    ck_tile::index_t stride_v_descale;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t nhead_stride_q_descale;
    ck_tile::index_t nhead_stride_k_descale;
    ck_tile::index_t nhead_stride_v_descale;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t batch_stride_q_descale;
    ck_tile::index_t batch_stride_k_descale;
    ck_tile::index_t batch_stride_v_descale;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t sink_size;
    ck_tile::index_t mask_type;
    ck_tile::index_t min_seqlen_q;

    float p_drop;
    bool s_randval;

    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;

    ck_tile::index_t block_scale_size_q;
    ck_tile::index_t block_scale_size_kv;
};

struct fmha_fwd_pagedkv_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    void* lse_ptr;
    void* o_ptr;

    void* block_table_ptr;
    ck_tile::index_t batch_stride_block_table; // only used if 'block_table_ptr' is not nullptr
    ck_tile::index_t page_block_size;          // only used if 'block_table_ptr' is not nullptr
    bool is_gappy; // differentiate seqstart_k_ptr usage. only used if 'block_table_ptr' is not
                   // nullptr.

    const void* cache_batch_idx;

    // the real seqlen_q & seqlen_k are decided by following:
    // batch mode: seqlen_q = kargs.seqlen_q
    //             seqlen_k = kargs.seqlen_k
    // group mode: seqlen_q = kargs.seqstart_q_ptr[b + 1] - kargs.seqstart_q_ptr[b]
    //             seqlen_k = kargs.seqstart_k_ptr[b + 1] - kargs.seqstart_k_ptr[b]
    //                      or kargs.seqlen_k_ptr[b]
    //
    // batch mode (kvcache):
    //             seqlen_q = kargs.seqlen_q
    //             seqlen_k = kargs.seqlen_k_ptr[b]
    // group mode (kvcache):
    //             seqlen_q = kargs.seqstart_q_ptr[b + 1] - kargs.seqstart_q_ptr[b]
    //
    //     when is_gappy=true:
    //             seqlen_k = kargs.seqlen_k_ptr[b]
    //             seqstart_k_ptr[b] now store local offset of each batch
    //
    //     when is_gappy=false:
    //             seqlen_k = kargs.seqstart_k_ptr[b + 1] - kargs.seqstart_k_ptr[b]
    //                      or kargs.seqlen_k_ptr[b]
    const void* seqstart_q_ptr;
    const void* seqstart_k_ptr;
    const void* seqlen_k_ptr;
    const void* sink_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    float scale_s;
    float scale_p;
    float scale_o;

    float logits_soft_cap;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t sink_size;
    ck_tile::index_t mask_type;
    ck_tile::index_t min_seqlen_q;
};

struct fmha_fwd_splitkv_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    void* lse_acc_ptr;
    void* o_acc_ptr;
    void* lse_ptr;
    void* o_ptr;

    void* block_table_ptr;
    ck_tile::index_t batch_stride_block_table; // only used if 'block_table_ptr' is not nullptr
    ck_tile::index_t page_block_size;          // only used if 'block_table_ptr' is not nullptr
    bool is_gappy; // differentiate seqstart_k_ptr usage. only used if 'block_table_ptr' is not
                   // nullptr.

    const void* cache_batch_idx;

    // the real seqlen_q & seqlen_k are decided by following:
    // batch mode: seqlen_q = kargs.seqlen_q
    //             seqlen_k = kargs.seqlen_k
    // group mode: seqlen_q = kargs.seqstart_q_ptr[b + 1] - kargs.seqstart_q_ptr[b]
    //             seqlen_k = kargs.seqstart_k_ptr[b + 1] - kargs.seqstart_k_ptr[b]
    //                      or kargs.seqlen_k_ptr[b]
    //
    // batch mode (kvcache):
    //             seqlen_q = kargs.seqlen_q
    //             seqlen_k = kargs.seqlen_k_ptr[b]
    // group mode (kvcache):
    //             seqlen_q = kargs.seqstart_q_ptr[b + 1] - kargs.seqstart_q_ptr[b]
    //
    //     when is_gappy=true:
    //             seqlen_k = kargs.seqlen_k_ptr[b]
    //             seqstart_k_ptr[b] now store local offset of each batch
    //
    //     when is_gappy=false:
    //             seqlen_k = kargs.seqstart_k_ptr[b + 1] - kargs.seqstart_k_ptr[b]
    //                      or kargs.seqlen_k_ptr[b]
    const void* seqstart_q_ptr;
    const void* seqstart_k_ptr;
    const void* seqlen_k_ptr;
    const void* sink_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;
    ck_tile::index_t num_splits;

    float scale_s;
    float scale_p;
    float scale_o;

    float logits_soft_cap;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_o_acc;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_lse_acc;
    ck_tile::index_t nhead_stride_o_acc;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_lse_acc;
    ck_tile::index_t batch_stride_o_acc;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t split_stride_lse_acc;
    ck_tile::index_t split_stride_o_acc;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t sink_size;
    ck_tile::index_t mask_type;
};

struct fmha_fwd_appendkv_args
{
    void* q_ptr;
    void* k_ptr;
    const void* knew_ptr;
    void* v_ptr;
    const void* vnew_ptr;

    const void* seqlen_k_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_knew;
    ck_tile::index_t batch;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    const void* rotary_cos_ptr; // only used if 'rotary_dim' > 0
    const void* rotary_sin_ptr; // only used if 'rotary_dim' > 0
    ck_tile::index_t rotary_dim;
    bool has_mask;

    void* block_table_ptr;
    ck_tile::index_t batch_stride_block_table; // only used if 'block_table_ptr' is not nullptr
    ck_tile::index_t page_block_size;          // only used if 'block_table_ptr' is not nullptr

    const void* cache_batch_idx; // only used if block_table_ptr is nullptr -> batch mode (kvcache)
    const void* sink_ptr;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_knew;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_vnew;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_knew;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_vnew;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_knew;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_vnew;
};

struct fmha_batch_prefill_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    const void* q_descale_ptr;
    const void* k_descale_ptr;
    const void* v_descale_ptr;
    void* rand_val_ptr;
    void* lse_ptr;
    void* o_ptr;

    // the real seqlen_q & seqlen_k are decided by following:
    // batch mode (kvcache):
    //             seqlen_q = kargs.seqlen_q
    //             seqlen_k = kargs.page_block_size * (kargs.kv_indptr[b + 1] - kargs.kv_indptr[b] -
    //             1) +
    //                        kargs.kv_last_page_lens[b]
    // group mode (kvcache):
    //             seqlen_q = kargs.seqstart_q_ptr[b + 1] - kargs.seqstart_q_ptr[b]
    //             seqlen_k = kargs.page_block_size * (kargs.kv_indptr[b + 1] - kargs.kv_indptr[b] -
    //             1) +
    //                        kargs.kv_last_page_lens[b]
    const void* seqstart_q_ptr;
    const void* sink_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    // KV cache page table fields (kv_lookup_table selects interpretation):
    // - SGLANG_PAGE_TABLE_1D:
    //   kv_indptr: prefix-sum [batch+1] into kv_page_indices
    //   kv_page_indices: 1D list of physical page ids, length = num_total_pages
    //   kv_last_page_lens: per-batch last page lengths [batch]
    // - VLLM_BLOCK_TABLE_2D:
    //   kv_page_indices: block_table [batch, max_blocks_per_seq] (2D)
    //   batch_stride_block_table: row stride for block_table
    //   seqlen_k_ptr: per-batch seqlen_k [batch]
    int32_t num_total_pages;          // total physical pages in KV cache (SGLang/vLLM)
    ck_tile::index_t page_block_size; // tokens per page (SGLang/vLLM)
    ck_tile::BlockAttentionKVCacheMemoryLayoutEnum
        kv_memory_layout;                                          // KV memory layout (SGLang/vLLM)
    ck_tile::BlockAttentionKVCacheLookupTableEnum kv_lookup_table; // lookup table layout selector
    void* kv_indptr;                           // SGLang: prefix-sum; vLLM: unused
    void* kv_page_indices;                     // SGLang: 1D page list; vLLM: block_table 2D
    void* kv_last_page_lens;                   // SGLang: last page lengths; vLLM: unused
    void* seqlen_k_ptr;                        // vLLM: per-batch seqlen_k; SGLang: unused
    ck_tile::index_t batch_stride_block_table; // vLLM: row stride; SGLang: unused

    float scale_s;
    float scale_p;
    float scale_o;

    float logits_soft_cap;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t sink_size;
    ck_tile::index_t mask_type;

    float p_drop;
    bool s_randval;

    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;

    // KV_BLOCKSCALE: per-page K/V descales (Q per-tensor, K/V per-page)
    // k_descale_ptr/v_descale_ptr are reused for KV_BLOCKSCALE mode:
    // k_descale_ptr: [num_block, num_kv_head] - points to k block descale
    // v_descale_ptr: [num_block, num_kv_head] - points to v block descale
    ck_tile::index_t nblock_stride_kv_block_descale = 0; // Stride along num_block dimension
    ck_tile::index_t nhead_stride_kv_block_descale  = 0; // Stride along num_kv_head dimension
};

// Selects the KV-cache load mode for a batch-prefill dispatch arm.
//   GLOBAL_LOAD_LDS: required when (a) the page is smaller than one K/V tile
//     so per-page SRD is impossible, AND (b) the total KV-pool byte size
//     exceeds INT32_MAX so SRD's 32-bit byte offset cannot address it.
//   BUFFER_LOAD: every other case — the SGPR-resident SRD path is fastest.
// Inputs are taken as plain integers so the helper has no template parameter
// and can be called from each codegen-emitted dispatcher arm with the arm's
// compile-time kN0 / element_bytes substituted as constants.
inline ck_tile::BlockAttentionKVCacheLoadModeEnum
fmha_batch_prefill_select_kv_load_mode(ck_tile::index_t page_block_size,
                                       ck_tile::index_t kN0,
                                       ck_tile::index_t num_total_pages,
                                       ck_tile::index_t batch_stride_k,
                                       ck_tile::index_t element_bytes)
{
    // Promote every operand to long_index_t so overflow is impossible regardless
    // of multiplication order. A bare `static_cast<long_index_t>(num_total_pages)
    // * batch_stride_k * element_bytes` only works because of left-to-right
    // associativity — a future reorder of the operands would silently truncate.
    const auto kv_pool_bytes = static_cast<ck_tile::long_index_t>(num_total_pages) *
                               static_cast<ck_tile::long_index_t>(batch_stride_k) *
                               static_cast<ck_tile::long_index_t>(element_bytes);
    return (page_block_size < kN0 && kv_pool_bytes > INT32_MAX)
               ? ck_tile::BlockAttentionKVCacheLoadModeEnum::GLOBAL_LOAD_LDS
               : ck_tile::BlockAttentionKVCacheLoadModeEnum::BUFFER_LOAD;
}

template <typename FmhaKernel>
auto fmha_fwd_create_kargs_and_grids(fmha_fwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaKernel::kIsGroupMode)
        {
            return FmhaKernel::MakeKargsImpl(args.q_ptr,
                                             args.k_ptr,
                                             args.v_ptr,
                                             args.bias_ptr,
                                             args.q_descale_ptr,
                                             args.k_descale_ptr,
                                             args.v_descale_ptr,
                                             args.rand_val_ptr,
                                             args.lse_ptr,
                                             args.o_ptr,
                                             args.seqstart_q_ptr,
                                             args.seqstart_k_ptr,
                                             args.seqlen_q_ptr,
                                             args.seqlen_k_ptr,
                                             args.block_scale_seqstart_q_ptr,
                                             args.block_scale_seqstart_k_ptr,
                                             args.seqstart_v_scale_ptr,
                                             args.hdim_q,
                                             args.hdim_v,
                                             args.nhead_q,
                                             args.nhead_q / args.nhead_k,
                                             args.scale_s,
                                             args.logits_soft_cap,
                                             args.stride_q,
                                             args.stride_k,
                                             args.stride_v,
                                             args.stride_bias,
                                             args.stride_randval,
                                             args.stride_o,
                                             args.stride_q_descale,
                                             args.stride_k_descale,
                                             args.stride_v_descale,
                                             args.nhead_stride_q,
                                             args.nhead_stride_k,
                                             args.nhead_stride_v,
                                             args.nhead_stride_bias,
                                             args.nhead_stride_randval,
                                             args.nhead_stride_lse,
                                             args.nhead_stride_o,
                                             args.nhead_stride_q_descale,
                                             args.nhead_stride_k_descale,
                                             args.nhead_stride_v_descale,
                                             args.window_size_left,
                                             args.window_size_right,
                                             args.sink_size,
                                             args.mask_type,
                                             args.min_seqlen_q,
                                             args.p_drop,
                                             args.s_randval,
                                             args.drop_seed_offset,
                                             args.block_scale_size_q,
                                             args.block_scale_size_kv,
                                             args.cu_seqlen_q_ptr,
                                             args.cu_seqlen_k_ptr,
                                             args.sink_ptr,
                                             args.num_head_q_total,
                                             args.head_start);
        }
        else
        { // create batch mode kernel arguments
            return FmhaKernel::MakeKargsImpl(args.q_ptr,
                                             args.k_ptr,
                                             args.v_ptr,
                                             args.bias_ptr,
                                             args.q_descale_ptr,
                                             args.k_descale_ptr,
                                             args.v_descale_ptr,
                                             args.rand_val_ptr,
                                             args.lse_ptr,
                                             args.o_ptr,
                                             args.seqlen_q,
                                             args.seqlen_k,
                                             args.hdim_q,
                                             args.hdim_v,
                                             args.nhead_q,
                                             args.nhead_q / args.nhead_k,
                                             args.scale_s,
                                             args.logits_soft_cap,
                                             args.stride_q,
                                             args.stride_k,
                                             args.stride_v,
                                             args.stride_bias,
                                             args.stride_randval,
                                             args.stride_o,
                                             args.stride_q_descale,
                                             args.stride_k_descale,
                                             args.stride_v_descale,
                                             args.nhead_stride_q,
                                             args.nhead_stride_k,
                                             args.nhead_stride_v,
                                             args.nhead_stride_bias,
                                             args.nhead_stride_randval,
                                             args.nhead_stride_lse,
                                             args.nhead_stride_o,
                                             args.nhead_stride_q_descale,
                                             args.nhead_stride_k_descale,
                                             args.nhead_stride_v_descale,
                                             args.batch_stride_q,
                                             args.batch_stride_k,
                                             args.batch_stride_v,
                                             args.batch_stride_bias,
                                             args.batch_stride_randval,
                                             args.batch_stride_lse,
                                             args.batch_stride_o,
                                             args.batch_stride_q_descale,
                                             args.batch_stride_k_descale,
                                             args.batch_stride_v_descale,
                                             args.window_size_left,
                                             args.window_size_right,
                                             args.sink_size,
                                             args.mask_type,
                                             args.p_drop,
                                             args.s_randval,
                                             args.drop_seed_offset,
                                             args.block_scale_size_q,
                                             args.block_scale_size_kv,
                                             args.cu_seqlen_q_ptr,
                                             args.cu_seqlen_k_ptr,
                                             args.sink_ptr,
                                             args.num_head_q_total,
                                             args.head_start);
        }
    }();

    if constexpr(FmhaKernel::kIsGroupMode)
    {
        dim3 grids = FmhaKernel::GridSize(
            args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v, args.seqlen_k_ptr != nullptr);
        return ck_tile::make_tuple(kargs, grids);
    }
    else
    {
        dim3 grids =
            FmhaKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v, false);
        return ck_tile::make_tuple(kargs, grids);
    }
}

template <typename FmhaKernel>
auto fmha_fwd_v3_create_kargs_and_grids(fmha_fwd_args args)
{
    /// NOTICE: This was borrowed from Aiter. Make sure the selected remap_opt setting truly
    /// maximizes the kernel's performance.
    int remap_opt = 2;
    if(args.mask_type != static_cast<int>(mask_enum::no_mask) &&
       ((args.nhead_q % 8 != 0) || (16384 < args.seqlen_q)))
    {
        if(65536 <= args.seqlen_q)
        {
            remap_opt = 0;
        }
        else
        {
            remap_opt = 1;
        }
    }

    auto kargs = [&] {
        if constexpr(FmhaKernel::kIsGroupMode)
        {
            return FmhaKernel::MakeKargs(args.q_ptr,
                                         args.k_ptr,
                                         args.v_ptr,
                                         args.q_descale_ptr,
                                         args.k_descale_ptr,
                                         args.v_descale_ptr,
                                         nullptr, // lse_ptr
                                         args.o_ptr,
                                         args.seqstart_q_ptr,
                                         args.seqstart_k_ptr,
                                         args.seqlen_q_ptr,
                                         args.seqlen_k_ptr,
                                         args.hdim_q,
                                         args.hdim_v,
                                         args.nhead_q,
                                         args.nhead_q / args.nhead_k,
                                         args.scale_s,
                                         args.logits_soft_cap,
                                         args.stride_q,
                                         args.stride_k,
                                         args.stride_v,
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         0, // nhead_stride_lse
                                         args.nhead_stride_o,
                                         args.window_size_left,
                                         args.window_size_right,
                                         args.mask_type,
                                         remap_opt,
                                         args.cu_seqlen_q_ptr,
                                         args.cu_seqlen_k_ptr);
        }
        else
        {
            return FmhaKernel::MakeKargs(args.q_ptr,
                                         args.k_ptr,
                                         args.v_ptr,
                                         args.q_descale_ptr,
                                         args.k_descale_ptr,
                                         args.v_descale_ptr,
                                         nullptr, // lse_ptr
                                         args.o_ptr,
                                         args.seqlen_q,
                                         args.seqlen_k,
                                         args.hdim_q,
                                         args.hdim_v,
                                         args.nhead_q,
                                         args.nhead_q / args.nhead_k,
                                         args.scale_s,
                                         args.logits_soft_cap,
                                         args.stride_q,
                                         args.stride_k,
                                         args.stride_v,
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         0, // nhead_stride_lse
                                         args.nhead_stride_o,
                                         args.batch_stride_q,
                                         args.batch_stride_k,
                                         args.batch_stride_v,
                                         0, // batch_stride_lse
                                         args.batch_stride_o,
                                         args.window_size_left,
                                         args.window_size_right,
                                         args.mask_type,
                                         remap_opt,
                                         args.cu_seqlen_q_ptr,
                                         args.cu_seqlen_k_ptr);
        }
    }();

    dim3 grids = FmhaKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v);

    return ck_tile::make_tuple(kargs, grids);
}

template <typename FmhaKernel>
auto fmha_fwd_pagedkv_create_kargs_and_grids(fmha_fwd_pagedkv_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaKernel::kIsGroupMode)
        {
            return FmhaKernel::MakeKargs(args.q_ptr,
                                         args.k_ptr,
                                         args.v_ptr,
                                         args.bias_ptr,
                                         args.lse_ptr,
                                         args.o_ptr,
                                         args.seqstart_q_ptr,
                                         args.seqstart_k_ptr,
                                         args.seqlen_k_ptr,
                                         args.hdim_q,
                                         args.hdim_v,
                                         args.nhead_q,
                                         args.nhead_q / args.nhead_k,
                                         args.block_table_ptr,
                                         args.batch_stride_block_table,
                                         args.page_block_size,
                                         args.is_gappy,
                                         args.scale_s,
                                         args.scale_p,
                                         args.scale_o,
                                         args.logits_soft_cap,
                                         args.stride_q,
                                         args.stride_k,
                                         args.stride_v,
                                         args.stride_bias,
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         args.nhead_stride_bias,
                                         args.nhead_stride_lse,
                                         args.nhead_stride_o,
                                         args.batch_stride_k,
                                         args.batch_stride_v,
                                         args.window_size_left,
                                         args.window_size_right,
                                         args.sink_size,
                                         args.mask_type,
                                         args.min_seqlen_q,
                                         args.sink_ptr);
        }
        else
        { // create batch mode kernel arguments
            return FmhaKernel::MakeKargs(args.q_ptr,
                                         args.k_ptr,
                                         args.v_ptr,
                                         args.bias_ptr,
                                         args.lse_ptr,
                                         args.o_ptr,
                                         args.seqlen_q,
                                         args.seqlen_k,
                                         args.seqlen_k_ptr,
                                         args.hdim_q,
                                         args.hdim_v,
                                         args.nhead_q,
                                         args.nhead_q / args.nhead_k,
                                         args.block_table_ptr,
                                         args.batch_stride_block_table,
                                         args.page_block_size,
                                         args.cache_batch_idx,
                                         args.scale_s,
                                         args.scale_p,
                                         args.scale_o,
                                         args.logits_soft_cap,
                                         args.stride_q,
                                         args.stride_k,
                                         args.stride_v,
                                         args.stride_bias,
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         args.nhead_stride_bias,
                                         args.nhead_stride_lse,
                                         args.nhead_stride_o,
                                         args.batch_stride_q,
                                         args.batch_stride_k,
                                         args.batch_stride_v,
                                         args.batch_stride_bias,
                                         args.batch_stride_lse,
                                         args.batch_stride_o,
                                         args.window_size_left,
                                         args.window_size_right,
                                         args.sink_size,
                                         args.mask_type,
                                         args.sink_ptr);
        }
    }();

    // FmhaKernel::PrintParameters(kargs, args.batch);
    if constexpr(FmhaKernel::kIsGroupMode)
    {
        dim3 grids = FmhaKernel::GridSize(
            args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v, args.seqlen_k_ptr != nullptr);
        return ck_tile::make_tuple(kargs, grids);
    }
    else
    {
        dim3 grids =
            FmhaKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v, false);
        return ck_tile::make_tuple(kargs, grids);
    }
}

template <typename Kernel>
auto fmha_fwd_splitkv_create_kargs_and_grids(fmha_fwd_splitkv_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(Kernel::kIsGroupMode)
        {
            return Kernel::MakeKargs(args.q_ptr,
                                     args.k_ptr,
                                     args.v_ptr,
                                     args.bias_ptr,
                                     args.lse_acc_ptr,
                                     args.o_acc_ptr,
                                     args.batch,
                                     args.seqstart_q_ptr,
                                     args.seqstart_k_ptr,
                                     args.seqlen_k_ptr,
                                     args.hdim_q,
                                     args.hdim_v,
                                     args.nhead_q,
                                     args.nhead_q / args.nhead_k,
                                     args.num_splits,
                                     args.block_table_ptr,
                                     args.batch_stride_block_table,
                                     args.page_block_size,
                                     args.is_gappy,
                                     args.scale_s,
                                     args.scale_p,
                                     args.logits_soft_cap,
                                     args.stride_q,
                                     args.stride_k,
                                     args.stride_v,
                                     args.stride_bias,
                                     args.stride_o_acc,
                                     args.nhead_stride_q,
                                     args.nhead_stride_k,
                                     args.nhead_stride_v,
                                     args.nhead_stride_bias,
                                     args.nhead_stride_lse_acc,
                                     args.nhead_stride_o_acc,
                                     args.batch_stride_k, // only used for paged-kvcache
                                     args.batch_stride_v, // only used for paged-kvcache
                                     args.split_stride_lse_acc,
                                     args.split_stride_o_acc,
                                     args.window_size_left,
                                     args.window_size_right,
                                     args.sink_size,
                                     args.mask_type,
                                     args.sink_ptr);
        }
        else
        { // create batch mode kernel arguments
            return Kernel::MakeKargs(args.q_ptr,
                                     args.k_ptr,
                                     args.v_ptr,
                                     args.bias_ptr,
                                     args.lse_acc_ptr,
                                     args.o_acc_ptr,
                                     args.batch,
                                     args.seqlen_q,
                                     args.seqlen_k,
                                     args.seqlen_k_ptr,
                                     args.hdim_q,
                                     args.hdim_v,
                                     args.nhead_q,
                                     args.nhead_q / args.nhead_k,
                                     args.num_splits,
                                     args.block_table_ptr,
                                     args.batch_stride_block_table,
                                     args.page_block_size,
                                     args.cache_batch_idx,
                                     args.scale_s,
                                     args.scale_p,
                                     args.logits_soft_cap,
                                     args.stride_q,
                                     args.stride_k,
                                     args.stride_v,
                                     args.stride_bias,
                                     args.stride_o_acc,
                                     args.nhead_stride_q,
                                     args.nhead_stride_k,
                                     args.nhead_stride_v,
                                     args.nhead_stride_bias,
                                     args.nhead_stride_lse_acc,
                                     args.nhead_stride_o_acc,
                                     args.batch_stride_q,
                                     args.batch_stride_k,
                                     args.batch_stride_v,
                                     args.batch_stride_bias,
                                     args.batch_stride_lse_acc,
                                     args.batch_stride_o_acc,
                                     args.split_stride_lse_acc,
                                     args.split_stride_o_acc,
                                     args.window_size_left,
                                     args.window_size_right,
                                     args.sink_size,
                                     args.mask_type,
                                     args.sink_ptr);
        }
    }();

    dim3 grids = Kernel::GridSize(
        args.batch, args.nhead_q, args.nhead_k, args.max_seqlen_q, args.hdim_v, args.num_splits);

    return ck_tile::make_tuple(kargs, grids);
}

template <typename Kernel>
auto fmha_fwd_splitkv_combine_create_kargs_and_grids(fmha_fwd_splitkv_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&] {
        // create group mode kernel argumentszs
        if constexpr(Kernel::kIsGroupMode)
        {
            return Kernel::MakeKargs(args.lse_acc_ptr,
                                     args.o_acc_ptr,
                                     args.lse_ptr,
                                     args.o_ptr,
                                     args.batch,
                                     args.seqstart_q_ptr,
                                     args.hdim_v,
                                     args.num_splits,
                                     args.scale_o,
                                     args.stride_o_acc,
                                     args.stride_o,
                                     args.nhead_stride_lse_acc,
                                     args.nhead_stride_o_acc,
                                     args.nhead_stride_lse,
                                     args.nhead_stride_o,
                                     args.split_stride_lse_acc,
                                     args.split_stride_o_acc);
        }
        else
        { // create batch mode kernel arguments
            return Kernel::MakeKargs(args.lse_acc_ptr,
                                     args.o_acc_ptr,
                                     args.lse_ptr,
                                     args.o_ptr,
                                     args.batch,
                                     args.seqlen_q,
                                     args.hdim_v,
                                     args.num_splits,
                                     args.scale_o,
                                     args.stride_o_acc,
                                     args.stride_o,
                                     args.nhead_stride_lse_acc,
                                     args.nhead_stride_o_acc,
                                     args.nhead_stride_lse,
                                     args.nhead_stride_o,
                                     args.batch_stride_lse_acc,
                                     args.batch_stride_o_acc,
                                     args.batch_stride_lse,
                                     args.batch_stride_o,
                                     args.split_stride_lse_acc,
                                     args.split_stride_o_acc);
        }
    }();

    dim3 grids = Kernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v);

    return ck_tile::make_tuple(kargs, grids);
}

template <typename Kernel>
auto fmha_fwd_appendkv_create_kargs_and_grids(fmha_fwd_appendkv_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = Kernel::MakeKargs(args.q_ptr,
                                   args.k_ptr,
                                   args.knew_ptr,
                                   args.v_ptr,
                                   args.vnew_ptr,
                                   args.seqlen_q,
                                   args.seqlen_k_ptr,
                                   args.seqlen_knew,
                                   args.hdim_q,
                                   args.hdim_v,
                                   args.nhead_q,
                                   args.nhead_q / args.nhead_k,
                                   args.rotary_cos_ptr,
                                   args.rotary_sin_ptr,
                                   args.rotary_dim,
                                   args.has_mask,
                                   args.block_table_ptr,
                                   args.batch_stride_block_table,
                                   args.page_block_size,
                                   args.cache_batch_idx,
                                   args.stride_q,
                                   args.stride_k,
                                   args.stride_knew,
                                   args.stride_v,
                                   args.stride_vnew,
                                   args.nhead_stride_q,
                                   args.nhead_stride_k,
                                   args.nhead_stride_knew,
                                   args.nhead_stride_v,
                                   args.nhead_stride_vnew,
                                   args.batch_stride_q,
                                   args.batch_stride_k,
                                   args.batch_stride_knew,
                                   args.batch_stride_v,
                                   args.batch_stride_vnew);

    dim3 grids = Kernel::GridSize(args.batch, args.nhead_q, args.seqlen_q, args.seqlen_knew);

    return ck_tile::make_tuple(kargs, grids);
}

template <typename FmhaKernel>
auto fmha_batch_prefill_create_kargs_and_grids(fmha_batch_prefill_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    using PageTableKargs            = typename FmhaKernel::PageBlockTableKargs;
    const PageTableKargs page_table = [&]() {
        if constexpr(FmhaKernel::kKVLookupTable ==
                     ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
        {
            return PageTableKargs{reinterpret_cast<const int32_t*>(args.kv_indptr),
                                  reinterpret_cast<const int32_t*>(args.kv_page_indices),
                                  reinterpret_cast<const int32_t*>(args.kv_last_page_lens)};
        }
        else
        {
            return PageTableKargs{reinterpret_cast<const int32_t*>(args.kv_page_indices),
                                  args.batch_stride_block_table,
                                  reinterpret_cast<const int32_t*>(args.seqlen_k_ptr)};
        }
    }();
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaKernel::kIsGroupMode)
        {
            return FmhaKernel::MakeKargs(args.q_ptr,
                                         args.k_ptr,
                                         args.v_ptr,
                                         args.bias_ptr,
                                         args.q_descale_ptr,
                                         args.k_descale_ptr,
                                         args.v_descale_ptr,
                                         args.rand_val_ptr,
                                         args.lse_ptr,
                                         args.o_ptr,
                                         args.seqstart_q_ptr,
                                         args.hdim_q,
                                         args.hdim_v,
                                         args.nhead_q,
                                         args.nhead_q / args.nhead_k,
                                         args.num_total_pages,
                                         args.page_block_size,
                                         page_table,
                                         args.scale_s,
                                         args.scale_p,
                                         args.scale_o,
                                         args.logits_soft_cap,
                                         args.stride_q,
                                         args.stride_k,
                                         args.stride_v,
                                         args.stride_bias,
                                         args.stride_randval,
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         args.nhead_stride_bias,
                                         args.nhead_stride_randval,
                                         args.nhead_stride_lse,
                                         args.nhead_stride_o,
                                         args.batch_stride_k,
                                         args.batch_stride_v,
                                         args.window_size_left,
                                         args.window_size_right,
                                         args.sink_size,
                                         args.mask_type,
                                         args.p_drop,
                                         args.s_randval,
                                         args.drop_seed_offset,
                                         args.sink_ptr,
                                         args.nblock_stride_kv_block_descale,
                                         args.nhead_stride_kv_block_descale);
        }
        else
        { // create batch mode kernel arguments
            return FmhaKernel::MakeKargs(args.q_ptr,
                                         args.k_ptr,
                                         args.v_ptr,
                                         args.bias_ptr,
                                         args.q_descale_ptr,
                                         args.k_descale_ptr,
                                         args.v_descale_ptr,
                                         args.rand_val_ptr,
                                         args.lse_ptr,
                                         args.o_ptr,
                                         args.seqlen_q,
                                         args.hdim_q,
                                         args.hdim_v,
                                         args.nhead_q,
                                         args.nhead_q / args.nhead_k,
                                         args.num_total_pages,
                                         args.page_block_size,
                                         page_table,
                                         args.scale_s,
                                         args.scale_p,
                                         args.scale_o,
                                         args.logits_soft_cap,
                                         args.stride_q,
                                         args.stride_k,
                                         args.stride_v,
                                         args.stride_bias,
                                         args.stride_randval,
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         args.nhead_stride_bias,
                                         args.nhead_stride_randval,
                                         args.nhead_stride_lse,
                                         args.nhead_stride_o,
                                         args.batch_stride_q,
                                         args.batch_stride_k,
                                         args.batch_stride_v,
                                         args.batch_stride_bias,
                                         args.batch_stride_randval,
                                         args.batch_stride_lse,
                                         args.batch_stride_o,
                                         args.window_size_left,
                                         args.window_size_right,
                                         args.sink_size,
                                         args.mask_type,
                                         args.p_drop,
                                         args.s_randval,
                                         args.drop_seed_offset,
                                         args.sink_ptr,
                                         args.nblock_stride_kv_block_descale,
                                         args.nhead_stride_kv_block_descale);
        }
    }();

    dim3 grids = FmhaKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v);
    return ck_tile::make_tuple(kargs, grids);
}

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          ck_tile::index_t kM0_,
          ck_tile::index_t kN0_,
          ck_tile::index_t kK0_,
          ck_tile::index_t kN1_,
          ck_tile::index_t kK1_,
          ck_tile::index_t kK0BlockLength_,
          bool kIsVLayoutRowMajor_,
          ck_tile::BlockFmhaPipelineEnum FmhaPipelineEnum_,
          bool kHasLogitsSoftCap_,
          typename FmhaMask_,
          ck_tile::BlockAttentionBiasEnum BiasEnum_,
          bool kStoreLse_,
          bool kHasDropout_,
          ck_tile::BlockAttentionQuantScaleEnum QScaleEnum_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_,
          bool kUseTrLoad_,
          bool kSkipMinSeqlenQ_ = false,
          bool kHasSink_        = false>
struct fmha_fwd_traits_
{
    static constexpr ck_tile::index_t HDim           = HDim_;
    using DataType                                   = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode               = kIsGroupMode_;
    static constexpr ck_tile::index_t kM0            = kM0_;
    static constexpr ck_tile::index_t kN0            = kN0_;
    static constexpr ck_tile::index_t kK0            = kK0_;
    static constexpr ck_tile::index_t kN1            = kN1_;
    static constexpr ck_tile::index_t kK1            = kK1_;
    static constexpr ck_tile::index_t kK0BlockLength = kK0BlockLength_;
    static constexpr bool kIsVLayoutRowMajor         = kIsVLayoutRowMajor_;
    static constexpr auto FmhaPipelineEnum           = FmhaPipelineEnum_;
    static constexpr bool kHasLogitsSoftCap          = kHasLogitsSoftCap_;
    using FmhaMask                                   = ck_tile::remove_cvref_t<FmhaMask_>;
    static constexpr auto BiasEnum                   = BiasEnum_;
    static constexpr bool kStoreLse                  = kStoreLse_;
    static constexpr bool kHasDropout                = kHasDropout_;
    static constexpr auto QScaleEnum                 = QScaleEnum_;
    static constexpr bool kPadS                      = kPadS_;
    static constexpr bool kPadSK                     = kPadSK_;
    static constexpr bool kPadD                      = kPadD_;
    static constexpr bool kPadDv                     = kPadDv_;
    static constexpr bool kUseTrLoad                 = kUseTrLoad_;
    static constexpr bool kSkipMinSeqlenQ            = kSkipMinSeqlenQ_;
    static constexpr bool kHasSink                   = kHasSink_;
};

template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          ck_tile::index_t kM0_,
          ck_tile::index_t kN0_,
          ck_tile::index_t kK0_,
          ck_tile::index_t kN1_,
          ck_tile::index_t kK1_,
          ck_tile::index_t kK0BlockLength_,
          bool kIsVLayoutRowMajor_,
          ck_tile::BlockFmhaPipelineEnum FmhaPipelineEnum_,
          bool kHasLogitsSoftCap_,
          typename FmhaMask_,
          ck_tile::BlockAttentionBiasEnum BiasEnum_,
          bool kStoreLse_,
          bool kHasDropout_,
          ck_tile::BlockAttentionQuantScaleEnum QScaleEnum_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_,
          bool kUseTrLoad_,
          bool kSkipMinSeqlenQ_            = false,
          bool kHasSink_                   = false,
          ck_tile::index_t kPageBlockSize_ = 1,
          ck_tile::BlockAttentionKVCacheMemoryLayoutEnum kKVMemoryLayout_ =
              ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT,
          ck_tile::BlockAttentionKVCacheLookupTableEnum kKVLookupTable_ =
              ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D,
          ck_tile::BlockAttentionKVCacheLoadModeEnum kKVLoadMode_ =
              ck_tile::BlockAttentionKVCacheLoadModeEnum::BUFFER_LOAD>
struct fmha_fwd_batch_prefill_traits_ : public fmha_fwd_traits_<HDim_,
                                                                DataType_,
                                                                kIsGroupMode_,
                                                                kM0_,
                                                                kN0_,
                                                                kK0_,
                                                                kN1_,
                                                                kK1_,
                                                                kK0BlockLength_,
                                                                kIsVLayoutRowMajor_,
                                                                FmhaPipelineEnum_,
                                                                kHasLogitsSoftCap_,
                                                                FmhaMask_,
                                                                BiasEnum_,
                                                                kStoreLse_,
                                                                kHasDropout_,
                                                                QScaleEnum_,
                                                                kPadS_,
                                                                kPadSK_,
                                                                kPadD_,
                                                                kPadDv_,
                                                                kUseTrLoad_,
                                                                kSkipMinSeqlenQ_,
                                                                kHasSink_>
{
    static constexpr auto kKVMemoryLayout            = kKVMemoryLayout_;
    static constexpr auto kKVLookupTable             = kKVLookupTable_;
    static constexpr ck_tile::index_t kPageBlockSize = kPageBlockSize_;
    static constexpr auto kKVLoadMode                = kKVLoadMode_;
    static_assert(kIsVLayoutRowMajor_, "Batch prefill only supports row-major V layout");
};

template <typename Traits_, typename Arch = void>
float fmha_fwd_(const ck_tile::stream_config&, fmha_fwd_args);

template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          ck_tile::index_t kM0_,
          ck_tile::index_t kN0_,
          ck_tile::index_t kK0_,
          ck_tile::index_t kN1_,
          ck_tile::index_t kK1_,
          ck_tile::index_t kK0BlockLength_,
          bool kIsVLayoutRowMajor_,
          ck_tile::BlockFmhaPipelineEnum FmhaPipelineEnum_,
          bool kHasLogitsSoftCap_,
          typename FmhaMask_,
          ck_tile::BlockAttentionBiasEnum BiasEnum_,
          bool kStoreLse_,
          bool kIsPagedKV_,
          bool kDoFp8StaticQuant_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_,
          bool kSkipMinSeqlenQ_ = false,
          bool kHasSink_        = false>
struct fmha_fwd_pagedkv_traits_
{
    static constexpr ck_tile::index_t HDim           = HDim_;
    using DataType                                   = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode               = kIsGroupMode_;
    static constexpr ck_tile::index_t kM0            = kM0_;
    static constexpr ck_tile::index_t kN0            = kN0_;
    static constexpr ck_tile::index_t kK0            = kK0_;
    static constexpr ck_tile::index_t kN1            = kN1_;
    static constexpr ck_tile::index_t kK1            = kK1_;
    static constexpr ck_tile::index_t kK0BlockLength = kK0BlockLength_;
    static constexpr bool kIsVLayoutRowMajor         = kIsVLayoutRowMajor_;
    static constexpr auto FmhaPipelineEnum           = FmhaPipelineEnum_;
    static constexpr bool kHasLogitsSoftCap          = kHasLogitsSoftCap_;
    using FmhaMask                                   = ck_tile::remove_cvref_t<FmhaMask_>;
    static constexpr auto BiasEnum                   = BiasEnum_;
    static constexpr bool kStoreLse                  = kStoreLse_;
    static constexpr bool kIsPagedKV                 = kIsPagedKV_;
    static constexpr bool kDoFp8StaticQuant          = kDoFp8StaticQuant_;
    static constexpr bool kPadS                      = kPadS_;
    static constexpr bool kPadSK                     = kPadSK_;
    static constexpr bool kPadD                      = kPadD_;
    static constexpr bool kPadDv                     = kPadDv_;
    static constexpr bool kSkipMinSeqlenQ            = kSkipMinSeqlenQ_;
    static constexpr bool kHasSink                   = kHasSink_;
};

template <typename Traits_, typename Arch = void>
float fmha_fwd_pagedkv_(const ck_tile::stream_config&, fmha_fwd_pagedkv_args);

template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          ck_tile::index_t kM0_,
          ck_tile::index_t kN0_,
          ck_tile::index_t kK0_,
          ck_tile::index_t kN1_,
          ck_tile::index_t kK1_,
          ck_tile::index_t kK0BlockLength_,
          bool kIsVLayoutRowMajor_,
          ck_tile::BlockFmhaPipelineEnum FmhaPipelineEnum_,
          bool kHasLogitsSoftCap_,
          typename FmhaMask_,
          ck_tile::BlockAttentionBiasEnum BiasEnum_,
          bool kStoreLse_,
          bool kDoFp8StaticQuant_,
          bool kIsPagedKV_,
          bool kHasSink_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_>
struct fmha_fwd_splitkv_traits_
{
    static constexpr ck_tile::index_t HDim           = HDim_;
    using DataType                                   = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode               = kIsGroupMode_;
    static constexpr ck_tile::index_t kM0            = kM0_;
    static constexpr ck_tile::index_t kN0            = kN0_;
    static constexpr ck_tile::index_t kK0            = kK0_;
    static constexpr ck_tile::index_t kN1            = kN1_;
    static constexpr ck_tile::index_t kK1            = kK1_;
    static constexpr ck_tile::index_t kK0BlockLength = kK0BlockLength_;
    static constexpr bool kIsVLayoutRowMajor         = kIsVLayoutRowMajor_;
    static constexpr auto FmhaPipelineEnum           = FmhaPipelineEnum_;
    static constexpr bool kHasLogitsSoftCap          = kHasLogitsSoftCap_;
    using FmhaMask                                   = ck_tile::remove_cvref_t<FmhaMask_>;
    static constexpr auto BiasEnum                   = BiasEnum_;
    static constexpr bool kStoreLse                  = kStoreLse_;
    static constexpr bool kDoFp8StaticQuant          = kDoFp8StaticQuant_;
    static constexpr bool kPadS                      = kPadS_;
    static constexpr bool kPadSK                     = kPadSK_;
    static constexpr bool kPadD                      = kPadD_;
    static constexpr bool kPadDv                     = kPadDv_;
    static constexpr bool kIsPagedKV                 = kIsPagedKV_;
    static constexpr bool kHasSink                   = kHasSink_;
};

template <typename Traits_, typename Arch = void>
void fmha_fwd_splitkv_oneshot_(const ck_tile::stream_config&, fmha_fwd_splitkv_args);

template <typename Traits_, typename Arch = void>
std::string fmha_fwd_splitkv_get_name_();

template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          ck_tile::index_t kN1_,
          bool kStoreLse_,
          bool kDoFp8StaticQuant_,
          bool kPadS_,
          bool kPadDv_>
struct fmha_fwd_splitkv_combine_traits_
{
    static constexpr ck_tile::index_t HDim  = HDim_;
    using DataType                          = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode      = kIsGroupMode_;
    static constexpr ck_tile::index_t kN1   = kN1_;
    static constexpr bool kStoreLse         = kStoreLse_;
    static constexpr bool kDoFp8StaticQuant = kDoFp8StaticQuant_;
    static constexpr bool kPadS             = kPadS_;
    static constexpr bool kPadDv            = kPadDv_;
};

template <typename Traits_, typename Arch = void>
void fmha_fwd_splitkv_combine_oneshot_(const ck_tile::stream_config&, fmha_fwd_splitkv_args);

template <typename Traits_, typename Arch = void>
std::string fmha_fwd_splitkv_combine_get_name_();

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <ck_tile::index_t HDim_,
          typename DataType_,
          ck_tile::index_t kTileSizeS_,
          ck_tile::index_t kTileSizeSk_,
          ck_tile::index_t kTileSizeD_,
          ck_tile::index_t kTileSizeDv_,
          bool kIsVLayoutRowMajor_,
          bool kPadS_,
          bool kPadSk_,
          bool kPadD_,
          bool kPadDv_,
          ck_tile::RotaryEmbeddingEnum RotaryEnum_,
          bool kIsPagedKV_>
struct fmha_fwd_appendkv_traits_
{
    static constexpr ck_tile::index_t HDim        = HDim_;
    using DataType                                = ck_tile::remove_cvref_t<DataType_>;
    static constexpr ck_tile::index_t kTileSizeS  = kTileSizeS_;
    static constexpr ck_tile::index_t kTileSizeSk = kTileSizeSk_;
    static constexpr ck_tile::index_t kTileSizeD  = kTileSizeD_;
    static constexpr ck_tile::index_t kTileSizeDv = kTileSizeDv_;
    static constexpr bool kIsVLayoutRowMajor      = kIsVLayoutRowMajor_;
    static constexpr bool kPadS                   = kPadS_;
    static constexpr bool kPadSk                  = kPadSk_;
    static constexpr bool kPadD                   = kPadD_;
    static constexpr bool kPadDv                  = kPadDv_;
    static constexpr auto RotaryEnum              = RotaryEnum_;
    static constexpr bool kIsPagedKV              = kIsPagedKV_;
};

template <typename Traits_, typename Arch = void>
float fmha_fwd_appendkv_(const ck_tile::stream_config&, fmha_fwd_appendkv_args);

template <typename Traits_, typename Arch = void>
float fmha_batch_prefill_(const ck_tile::stream_config&, fmha_batch_prefill_args);

// This is the public API, will be generated by script
struct fmha_fwd_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    bool has_logits_soft_cap;
    mask_enum mask_type;
    bias_enum bias_type; // 0:no bias, 1:elementwise bias, 2:alibi. sync with BlockAttentionBiasEnum
    bool has_lse;
    bool has_dropout;
    quant_scale_enum qscale_type;
    bool skip_min_seqlen_q = false;
    bool has_sink          = false;
    // TODO: padding check is inside this api
};
float fmha_fwd(fmha_fwd_traits, fmha_fwd_args, const ck_tile::stream_config&);

struct fmha_fwd_pagedkv_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    bool has_logits_soft_cap;
    mask_enum mask_type;
    bias_enum bias_type; // 0:no bias, 1:elementwise bias, 2:alibi. sync with BlockAttentionBiasEnum
    bool has_lse             = false;
    bool use_pagedkv         = true;
    bool do_fp8_static_quant = false;
    bool skip_min_seqlen_q   = false;
    bool has_sink            = false;
    // TODO: padding check is inside this api
};

float fmha_fwd_pagedkv(fmha_fwd_pagedkv_traits&,
                       fmha_fwd_pagedkv_args&,
                       const ck_tile::stream_config&);

struct fmha_fwd_splitkv_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    bool has_logits_soft_cap;
    mask_enum mask_type;
    bias_enum bias_type; // 0:no bias, 1:elementwise bias, 2:alibi. sync with BlockAttentionBiasEnum
    bool has_lse;
    bool do_fp8_static_quant = false;
    bool has_sink            = false;
    // TODO: padding check is inside this api
};
float fmha_fwd_splitkv(fmha_fwd_splitkv_traits,
                       fmha_fwd_splitkv_args,
                       const ck_tile::stream_config&);

struct fmha_fwd_appendkv_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_v_rowmajor;
    rope_enum rope_type;
};
float fmha_fwd_appendkv(fmha_fwd_appendkv_traits,
                        fmha_fwd_appendkv_args,
                        const ck_tile::stream_config&);

struct fmha_batch_prefill_traits : public fmha_fwd_traits
{
    ck_tile::BlockAttentionKVCacheMemoryLayoutEnum kv_memory_layout =
        ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    ck_tile::BlockAttentionKVCacheLookupTableEnum kv_lookup_table =
        ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D;
    int page_size = 1;
};

float fmha_batch_prefill(fmha_batch_prefill_traits,
                         fmha_batch_prefill_args,
                         const ck_tile::stream_config&);
