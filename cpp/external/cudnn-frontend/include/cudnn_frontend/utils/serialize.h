#pragma once

#include "../graph_properties.h"
#include "../graph_helpers.h"

namespace cudnn_frontend::graph {
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
NLOHMANN_JSON_SERIALIZE_ENUM(BN_finalize_attributes::input_names,
                             {
                                 {BN_finalize_attributes::input_names::SUM, "SUM"},
                                 {BN_finalize_attributes::input_names::SQ_SUM, "SQ_SUM"},
                                 {BN_finalize_attributes::input_names::SCALE, "SCALE"},
                                 {BN_finalize_attributes::input_names::BIAS, "BIAS"},
                                 {BN_finalize_attributes::input_names::EPSILON, "EPSILON"},
                                 {BN_finalize_attributes::input_names::ACCUM_COUNT, "ACCUM_COUNT"},
                                 {BN_finalize_attributes::input_names::PREV_RUNNING_MEAN, "PREV_RUNNING_MEAN"},
                                 {BN_finalize_attributes::input_names::PREV_RUNNING_VAR, "PREV_RUNNING_VAR"},
                                 {BN_finalize_attributes::input_names::MOMENTUM, "MOMENTUM"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(BN_finalize_attributes::output_names,
                             {
                                 {BN_finalize_attributes::output_names::EQ_SCALE, "EQ_SCALE"},
                                 {BN_finalize_attributes::output_names::EQ_BIAS, "EQ_BIAS"},
                                 {BN_finalize_attributes::output_names::MEAN, "MEAN"},
                                 {BN_finalize_attributes::output_names::INV_VARIANCE, "INV_VARIANCE"},
                                 {BN_finalize_attributes::output_names::NEXT_RUNNING_MEAN, "NEXT_RUNNING_MEAN"},
                                 {BN_finalize_attributes::output_names::NEXT_RUNNING_VAR, "NEXT_RUNNING_VAR"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Batchnorm_attributes::input_names,
                             {
                                 {Batchnorm_attributes::input_names::X, "X"},
                                 {Batchnorm_attributes::input_names::SCALE, "SCALE"},
                                 {Batchnorm_attributes::input_names::BIAS, "BIAS"},
                                 {Batchnorm_attributes::input_names::EPSILON, "EPSILON"},
                                 {Batchnorm_attributes::input_names::PREV_RUNNING_MEAN, "PREV_RUNNING_MEAN"},
                                 {Batchnorm_attributes::input_names::PREV_RUNNING_VAR, "PREV_RUNNING_VAR"},
                                 {Batchnorm_attributes::input_names::MOMENTUM, "MOMENTUM"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Batchnorm_attributes::output_names,
                             {
                                 {Batchnorm_attributes::output_names::Y, "Y"},
                                 {Batchnorm_attributes::output_names::MEAN, "MEAN"},
                                 {Batchnorm_attributes::output_names::INV_VARIANCE, "INV_VARIANCE"},
                                 {Batchnorm_attributes::output_names::NEXT_RUNNING_MEAN, "NEXT_RUNNING_MEAN"},
                                 {Batchnorm_attributes::output_names::NEXT_RUNNING_VAR, "NEXT_RUNNING_VAR"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Batchnorm_backward_attributes::input_names,
                             {
                                 {Batchnorm_backward_attributes::input_names::DY, "DY"},
                                 {Batchnorm_backward_attributes::input_names::X, "X"},
                                 {Batchnorm_backward_attributes::input_names::SCALE, "SCALE"},
                                 {Batchnorm_backward_attributes::input_names::MEAN, "MEAN"},
                                 {Batchnorm_backward_attributes::input_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Batchnorm_backward_attributes::output_names,
                             {
                                 {Batchnorm_backward_attributes::output_names::DX, "DX"},
                                 {Batchnorm_backward_attributes::output_names::DSCALE, "DSCALE"},
                                 {Batchnorm_backward_attributes::output_names::DBIAS, "DBIAS"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Batchnorm_inference_attributes::input_names,
                             {
                                 {Batchnorm_inference_attributes::input_names::X, "X"},
                                 {Batchnorm_inference_attributes::input_names::SCALE, "SCALE"},
                                 {Batchnorm_inference_attributes::input_names::BIAS, "BIAS"},
                                 {Batchnorm_inference_attributes::input_names::MEAN, "MEAN"},
                                 {Batchnorm_inference_attributes::input_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Batchnorm_inference_attributes::output_names,
                             {{Batchnorm_inference_attributes::output_names::Y, "Y"}})

NLOHMANN_JSON_SERIALIZE_ENUM(Conv_dgrad_attributes::input_names,
                             {
                                 {Conv_dgrad_attributes::input_names::W, "W"},
                                 {Conv_dgrad_attributes::input_names::DY, "DY"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Conv_dgrad_attributes::output_names,
                             {
                                 {Conv_dgrad_attributes::output_names::DX, "DX"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Conv_fprop_attributes::input_names,
                             {
                                 {Conv_fprop_attributes::input_names::X, "X"},
                                 {Conv_fprop_attributes::input_names::W, "W"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Conv_fprop_attributes::output_names,
                             {
                                 {Conv_fprop_attributes::output_names::Y, "Y"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Conv_wgrad_attributes::input_names,
                             {
                                 {Conv_wgrad_attributes::input_names::DY, "DY"},
                                 {Conv_wgrad_attributes::input_names::X, "X"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Conv_wgrad_attributes::output_names,
                             {
                                 {Conv_wgrad_attributes::output_names::DW, "DW"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(DBN_weight_attributes::input_names,
                             {
                                 {DBN_weight_attributes::input_names::DY, "DY"},
                                 {DBN_weight_attributes::input_names::X, "X"},
                                 {DBN_weight_attributes::input_names::SCALE, "SCALE"},
                                 {DBN_weight_attributes::input_names::MEAN, "MEAN"},
                                 {DBN_weight_attributes::input_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(DBN_weight_attributes::output_names,
                             {
                                 {DBN_weight_attributes::output_names::DSCALE, "DSCALE"},
                                 {DBN_weight_attributes::output_names::DBIAS, "DBIAS"},
                                 {DBN_weight_attributes::output_names::EQ_BIAS, "EQ_BIAS"},
                                 {DBN_weight_attributes::output_names::EQ_SCALE_DY, "EQ_SCALE_DY"},
                                 {DBN_weight_attributes::output_names::EQ_SCALE_X, "EQ_SCALE_X"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Genstats_attributes::input_names,
                             {
                                 {Genstats_attributes::input_names::X, "X"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Genstats_attributes::output_names,
                             {
                                 {Genstats_attributes::output_names::SUM, "SUM"},
                                 {Genstats_attributes::output_names::SQ_SUM, "SQ_SUM"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Instancenorm_attributes::input_names,
                             {
                                 {Instancenorm_attributes::input_names::X, "X"},
                                 {Instancenorm_attributes::input_names::SCALE, "SCALE"},
                                 {Instancenorm_attributes::input_names::BIAS, "BIAS"},
                                 {Instancenorm_attributes::input_names::EPSILON, "EPSILON"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Instancenorm_attributes::output_names,
                             {
                                 {Instancenorm_attributes::output_names::Y, "Y"},
                                 {Instancenorm_attributes::output_names::MEAN, "MEAN"},
                                 {Instancenorm_attributes::output_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Instancenorm_backward_attributes::input_names,
                             {
                                 {Instancenorm_backward_attributes::input_names::DY, "DY"},
                                 {Instancenorm_backward_attributes::input_names::X, "X"},
                                 {Instancenorm_backward_attributes::input_names::SCALE, "SCALE"},
                                 {Instancenorm_backward_attributes::input_names::MEAN, "MEAN"},
                                 {Instancenorm_backward_attributes::input_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Instancenorm_backward_attributes::output_names,
                             {
                                 {Instancenorm_backward_attributes::output_names::DX, "DX"},
                                 {Instancenorm_backward_attributes::output_names::DSCALE, "DSCALE"},
                                 {Instancenorm_backward_attributes::output_names::DBIAS, "DBIAS"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Layernorm_attributes::input_names,
                             {
                                 {Layernorm_attributes::input_names::X, "X"},
                                 {Layernorm_attributes::input_names::SCALE, "SCALE"},
                                 {Layernorm_attributes::input_names::BIAS, "BIAS"},
                                 {Layernorm_attributes::input_names::EPSILON, "EPSILON"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Layernorm_attributes::output_names,
                             {
                                 {Layernorm_attributes::output_names::Y, "Y"},
                                 {Layernorm_attributes::output_names::MEAN, "MEAN"},
                                 {Layernorm_attributes::output_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Layernorm_backward_attributes::input_names,
                             {
                                 {Layernorm_backward_attributes::input_names::DY, "DY"},
                                 {Layernorm_backward_attributes::input_names::X, "X"},
                                 {Layernorm_backward_attributes::input_names::SCALE, "SCALE"},
                                 {Layernorm_backward_attributes::input_names::MEAN, "MEAN"},
                                 {Layernorm_backward_attributes::input_names::INV_VARIANCE, "INV_VARIANCE"},
                                 {Layernorm_backward_attributes::input_names::EPSILON, "EPSILON"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Layernorm_backward_attributes::output_names,
                             {
                                 {Layernorm_backward_attributes::output_names::DX, "DX"},
                                 {Layernorm_backward_attributes::output_names::DSCALE, "DSCALE"},
                                 {Layernorm_backward_attributes::output_names::DBIAS, "DBIAS"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(AdaLayernorm_attributes::input_names,
                             {
                                 {AdaLayernorm_attributes::input_names::X, "X"},
                                 {AdaLayernorm_attributes::input_names::SCALE, "SCALE"},
                                 {AdaLayernorm_attributes::input_names::BIAS, "BIAS"},
                                 {AdaLayernorm_attributes::input_names::EPSILON, "EPSILON"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(AdaLayernorm_attributes::output_names,
                             {
                                 {AdaLayernorm_attributes::output_names::Y, "Y"},
                                 {AdaLayernorm_attributes::output_names::MEAN, "MEAN"},
                                 {AdaLayernorm_attributes::output_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(AdaLayernorm_backward_attributes::input_names,
                             {
                                 {AdaLayernorm_backward_attributes::input_names::DY, "DY"},
                                 {AdaLayernorm_backward_attributes::input_names::X, "X"},
                                 {AdaLayernorm_backward_attributes::input_names::SCALE, "SCALE"},
                                 {AdaLayernorm_backward_attributes::input_names::MEAN, "MEAN"},
                                 {AdaLayernorm_backward_attributes::input_names::INV_VARIANCE, "INV_VARIANCE"},
                                 {AdaLayernorm_backward_attributes::input_names::EPSILON, "EPSILON"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(AdaLayernorm_backward_attributes::output_names,
                             {
                                 {AdaLayernorm_backward_attributes::output_names::DX, "DX"},
                                 {AdaLayernorm_backward_attributes::output_names::DSCALE, "DSCALE"},
                                 {AdaLayernorm_backward_attributes::output_names::DBIAS, "DBIAS"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Matmul_attributes::input_names,
                             {
                                 {Matmul_attributes::input_names::A, "A"},
                                 {Matmul_attributes::input_names::B, "B"},
                                 {Matmul_attributes::input_names::M_override, "M_override"},
                                 {Matmul_attributes::input_names::N_override, "N_override"},
                                 {Matmul_attributes::input_names::K_override, "K_override"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Matmul_attributes::output_names,
                             {
                                 {Matmul_attributes::output_names::C, "C"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Matmul_fp8_attributes::input_names,
                             {
                                 {Matmul_fp8_attributes::input_names::A, "A"},
                                 {Matmul_fp8_attributes::input_names::B, "B"},
                                 {Matmul_fp8_attributes::input_names::Descale_A, "Descale_A"},
                                 {Matmul_fp8_attributes::input_names::Descale_B, "Descale_B"},
                                 {Matmul_fp8_attributes::input_names::M_override, "M_override"},
                                 {Matmul_fp8_attributes::input_names::N_override, "N_override"},
                                 {Matmul_fp8_attributes::input_names::K_override, "K_override"},
                                 {Matmul_fp8_attributes::input_names::Scale_C, "Scale_C"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Matmul_fp8_attributes::output_names,
                             {
                                 {Matmul_fp8_attributes::output_names::C, "C"},
                                 {Matmul_fp8_attributes::output_names::Amax_C, "Amax_C"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Pointwise_attributes::input_names,
                             {
                                 {Pointwise_attributes::input_names::IN_0, "IN_0"},
                                 {Pointwise_attributes::input_names::IN_1, "IN_1"},
                                 {Pointwise_attributes::input_names::IN_2, "IN_2"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Pointwise_attributes::output_names,
                             {
                                 {Pointwise_attributes::output_names::OUT_0, "OUT_0"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Reduction_attributes::input_names,
                             {
                                 {Reduction_attributes::input_names::X, "X"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Reduction_attributes::output_names, {{Reduction_attributes::output_names::Y, "Y"}})

NLOHMANN_JSON_SERIALIZE_ENUM(Resample_attributes::input_names,
                             {
                                 {Resample_attributes::input_names::X, "X"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Resample_attributes::output_names,
                             {{Resample_attributes::output_names::Y, "Y"},
                              {Resample_attributes::output_names::Index, "Index"}})

NLOHMANN_JSON_SERIALIZE_ENUM(Reshape_attributes::input_names,
                             {
                                 {Reshape_attributes::input_names::X, "X"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Reshape_attributes::output_names, {{Reshape_attributes::output_names::Y, "Y"}})

NLOHMANN_JSON_SERIALIZE_ENUM(Transpose_attributes::input_names,
                             {
                                 {Transpose_attributes::input_names::X, "X"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Transpose_attributes::output_names, {{Transpose_attributes::output_names::Y, "Y"}})

NLOHMANN_JSON_SERIALIZE_ENUM(Rmsnorm_attributes::input_names,
                             {
                                 {Rmsnorm_attributes::input_names::X, "X"},
                                 {Rmsnorm_attributes::input_names::SCALE, "SCALE"},
                                 {Rmsnorm_attributes::input_names::BIAS, "BIAS"},
                                 {Rmsnorm_attributes::input_names::EPSILON, "EPSILON"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Rmsnorm_attributes::output_names,
                             {
                                 {Rmsnorm_attributes::output_names::Y, "Y"},
                                 {Rmsnorm_attributes::output_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Rmsnorm_backward_attributes::input_names,
                             {
                                 {Rmsnorm_backward_attributes::input_names::DY, "DY"},
                                 {Rmsnorm_backward_attributes::input_names::X, "X"},
                                 {Rmsnorm_backward_attributes::input_names::SCALE, "SCALE"},
                                 {Rmsnorm_backward_attributes::input_names::INV_VARIANCE, "INV_VARIANCE"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Rmsnorm_backward_attributes::output_names,
                             {
                                 {Rmsnorm_backward_attributes::output_names::DX, "DX"},
                                 {Rmsnorm_backward_attributes::output_names::DSCALE, "DSCALE"},
                                 {Rmsnorm_backward_attributes::output_names::DBIAS, "DBIAS"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Rng_attributes::input_names,
                             {
                                 {Rng_attributes::input_names::Seed, "Seed"},
                                 {Rng_attributes::input_names::Offset, "Offset"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Rng_attributes::output_names, {{Rng_attributes::output_names::Y, "Y"}})

NLOHMANN_JSON_SERIALIZE_ENUM(PagedCacheLoad_attributes::input_names,
                             {
                                 {PagedCacheLoad_attributes::input_names::container, "container"},
                                 {PagedCacheLoad_attributes::input_names::seqLen, "seqLen"},
                                 {PagedCacheLoad_attributes::input_names::pageTable, "pageTable"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(PagedCacheLoad_attributes::output_names,
                             {
                                 {PagedCacheLoad_attributes::output_names::yOut, "yOut"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(SDPA_attributes::input_names,
                             {
                                 {SDPA_attributes::input_names::Q, "Q"},
                                 {SDPA_attributes::input_names::K, "K"},
                                 {SDPA_attributes::input_names::V, "V"},
                                 {SDPA_attributes::input_names::Attn_scale, "Attn_scale"},
                                 {SDPA_attributes::input_names::Bias, "Bias"},
                                 {SDPA_attributes::input_names::SEQ_LEN_Q, "SEQ_LEN_Q"},
                                 {SDPA_attributes::input_names::SEQ_LEN_KV, "SEQ_LEN_KV"},
                                 {SDPA_attributes::input_names::Seed, "Seed"},
                                 {SDPA_attributes::input_names::Offset, "Offset"},
                                 {SDPA_attributes::input_names::Dropout_mask, "Dropout_mask"},
                                 {SDPA_attributes::input_names::Dropout_scale, "Dropout_scale"},
                                 {SDPA_attributes::input_names::Page_table_K, "Page_table_K"},
                                 {SDPA_attributes::input_names::Page_table_V, "Page_table_V"},
                                 {SDPA_attributes::input_names::Block_mask, "Block_mask"},
                                 // FP8-specific inputs
                                 {SDPA_attributes::input_names::Descale_Q, "Descale_Q"},
                                 {SDPA_attributes::input_names::Descale_K, "Descale_K"},
                                 {SDPA_attributes::input_names::Descale_V, "Descale_V"},
                                 {SDPA_attributes::input_names::Descale_S, "Descale_S"},
                                 {SDPA_attributes::input_names::Scale_S, "Scale_S"},
                                 {SDPA_attributes::input_names::Scale_O, "Scale_O"},
                                 {SDPA_attributes::input_names::SINK_TOKEN, "SINK_TOKEN"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(SDPA_attributes::output_names,
                             {{SDPA_attributes::output_names::O, "O"},
                              {SDPA_attributes::output_names::Stats, "Stats"},
                              {SDPA_attributes::output_names::Max, "Max"},
                              {SDPA_attributes::output_names::Sum_exp, "Sum_exp"},
                              {SDPA_attributes::output_names::RNG_DUMP, "RNG_DUMP"},
                              {SDPA_attributes::output_names::Amax_S, "Amax_S"},
                              {SDPA_attributes::output_names::Amax_O, "Amax_O"}})

NLOHMANN_JSON_SERIALIZE_ENUM(SDPA_backward_attributes::input_names,
                             {
                                 {SDPA_backward_attributes::input_names::Q, "Q"},
                                 {SDPA_backward_attributes::input_names::K, "K"},
                                 {SDPA_backward_attributes::input_names::V, "V"},
                                 {SDPA_backward_attributes::input_names::O, "O"},
                                 {SDPA_backward_attributes::input_names::dO, "dO"},
                                 {SDPA_backward_attributes::input_names::Stats, "Stats"},
                                 {SDPA_backward_attributes::input_names::Attn_scale, "Attn_scale"},
                                 {SDPA_backward_attributes::input_names::Bias, "Bias"},
                                 {SDPA_backward_attributes::input_names::SEQ_LEN_Q, "SEQ_LEN_Q"},
                                 {SDPA_backward_attributes::input_names::SEQ_LEN_KV, "SEQ_LEN_KV"},
                                 {SDPA_backward_attributes::input_names::Seed, "Seed"},
                                 {SDPA_backward_attributes::input_names::Offset, "Offset"},
                                 {SDPA_backward_attributes::input_names::Dropout_mask, "Dropout_mask"},
                                 {SDPA_backward_attributes::input_names::Dropout_scale, "Dropout_scale"},
                                 {SDPA_backward_attributes::input_names::Dropout_scale_inv, "Dropout_scale_inv"},
                                 {SDPA_backward_attributes::input_names::SINK_TOKEN, "SINK_TOKEN"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(SDPA_backward_attributes::output_names,
                             {
                                 {SDPA_backward_attributes::output_names::dQ, "dQ"},
                                 {SDPA_backward_attributes::output_names::dK, "dK"},
                                 {SDPA_backward_attributes::output_names::dV, "dV"},
                                 {SDPA_backward_attributes::output_names::dBias, "dBias"},
                                 {SDPA_backward_attributes::output_names::RNG_DUMP, "RNG_DUMP"},
                                 {SDPA_backward_attributes::output_names::DSINK_TOKEN, "DSINK_TOKEN"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Slice_attributes::output_names,
                             {
                                 {Slice_attributes::output_names::Y, "Y"},
                             })
NLOHMANN_JSON_SERIALIZE_ENUM(Slice_attributes::input_names,
                             {
                                 {Slice_attributes::input_names::X, "X"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(SDPA_fp8_backward_attributes::input_names,
                             {
                                 {SDPA_fp8_backward_attributes::input_names::Q, "Q"},
                                 {SDPA_fp8_backward_attributes::input_names::Q_T, "Q_T"},
                                 {SDPA_fp8_backward_attributes::input_names::K, "K"},
                                 {SDPA_fp8_backward_attributes::input_names::K_T, "K_T"},
                                 {SDPA_fp8_backward_attributes::input_names::V, "V"},
                                 {SDPA_fp8_backward_attributes::input_names::O, "O"},
                                 {SDPA_fp8_backward_attributes::input_names::dO, "dO"},
                                 {SDPA_fp8_backward_attributes::input_names::dO_T, "dO_T"},
                                 {SDPA_fp8_backward_attributes::input_names::dO_f16, "dO_f16"},
                                 {SDPA_fp8_backward_attributes::input_names::Stats, "Stats"},
                                 {SDPA_fp8_backward_attributes::input_names::Attn_scale, "Attn_scale"},
                                 {SDPA_fp8_backward_attributes::input_names::Bias, "Bias"},
                                 {SDPA_fp8_backward_attributes::input_names::SEQ_LEN_Q, "SEQ_LEN_Q"},
                                 {SDPA_fp8_backward_attributes::input_names::SEQ_LEN_KV, "SEQ_LEN_KV"},
                                 {SDPA_fp8_backward_attributes::input_names::Seed, "Seed"},
                                 {SDPA_fp8_backward_attributes::input_names::Offset, "Offset"},
                                 {SDPA_fp8_backward_attributes::input_names::Dropout_mask, "Dropout_mask"},
                                 {SDPA_fp8_backward_attributes::input_names::Dropout_scale, "Dropout_scale"},
                                 {SDPA_fp8_backward_attributes::input_names::Dropout_scale_inv, "Dropout_scale_inv"},

                                 {SDPA_fp8_backward_attributes::input_names::Descale_Q, "Descale_Q"},
                                 {SDPA_fp8_backward_attributes::input_names::Descale_K, "Descale_K"},
                                 {SDPA_fp8_backward_attributes::input_names::Descale_V, "Descale_V"},
                                 {SDPA_fp8_backward_attributes::input_names::Descale_O, "Descale_O"},
                                 {SDPA_fp8_backward_attributes::input_names::Descale_dO, "Descale_dO"},
                                 {SDPA_fp8_backward_attributes::input_names::Descale_dO_T, "Descale_dO_T"},
                                 {SDPA_fp8_backward_attributes::input_names::Descale_K_T, "Descale_K_T"},
                                 {SDPA_fp8_backward_attributes::input_names::Descale_Q_T, "Descale_Q_T"},
                                 {SDPA_fp8_backward_attributes::input_names::Descale_S, "Descale_S"},
                                 {SDPA_fp8_backward_attributes::input_names::Descale_dP, "Descale_dP"},
                                 {SDPA_fp8_backward_attributes::input_names::Scale_dQ, "Scale_dQ"},
                                 {SDPA_fp8_backward_attributes::input_names::Scale_dK, "Scale_dK"},
                                 {SDPA_fp8_backward_attributes::input_names::Scale_dV, "Scale_dV"},
                                 {SDPA_fp8_backward_attributes::input_names::Scale_S, "Scale_S"},
                                 {SDPA_fp8_backward_attributes::input_names::Scale_dP, "Scale_dP"},
                                 {SDPA_fp8_backward_attributes::input_names::SINK_TOKEN, "SINK_TOKEN"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(SDPA_fp8_backward_attributes::output_names,
                             {
                                 {SDPA_fp8_backward_attributes::output_names::dQ, "dQ"},
                                 {SDPA_fp8_backward_attributes::output_names::dK, "dK"},
                                 {SDPA_fp8_backward_attributes::output_names::dV, "dV"},
                                 {SDPA_fp8_backward_attributes::output_names::Amax_dQ, "Amax_dQ"},
                                 {SDPA_fp8_backward_attributes::output_names::Amax_dK, "Amax_dK"},
                                 {SDPA_fp8_backward_attributes::output_names::Amax_dV, "Amax_dV"},
                                 {SDPA_fp8_backward_attributes::output_names::Amax_dP, "Amax_dP"},
                                 {SDPA_fp8_backward_attributes::output_names::DSINK_TOKEN, "DSINK_TOKEN"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(RoPE_attributes::input_names,
                             {
                                 {RoPE_attributes::input_names::INPUT, "INPUT"},
                                 {RoPE_attributes::input_names::FREQS, "FREQS"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(RoPE_attributes::output_names,
                             {
                                 {RoPE_attributes::output_names::OUTPUT, "OUTPUT"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(RoPE_backward_attributes::input_names,
                             {
                                 {RoPE_backward_attributes::input_names::DY, "DY"},
                                 {RoPE_backward_attributes::input_names::FREQS, "FREQS"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(RoPE_backward_attributes::output_names,
                             {
                                 {RoPE_backward_attributes::output_names::DX, "DX"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Softmax_attributes::input_names,
                             {
                                 {Softmax_attributes::input_names::P, "P"},
                                 {Softmax_attributes::input_names::SINK, "SINK"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Softmax_attributes::output_names,
                             {
                                 {Softmax_attributes::output_names::S, "S"},
                                 {Softmax_attributes::output_names::Stats, "Stats"},
                                 {Softmax_attributes::output_names::Max, "Max"},
                                 {Softmax_attributes::output_names::Sum_exp, "Sum_exp"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(DiagonalBandMask_attributes::input_names,
                             {
                                 {DiagonalBandMask_attributes::input_names::X, "X"},
                                 {DiagonalBandMask_attributes::input_names::SEQ_LEN_Q, "SEQ_LEN_Q"},
                                 {DiagonalBandMask_attributes::input_names::SEQ_LEN_KV, "SEQ_LEN_KV"},
                                 {DiagonalBandMask_attributes::input_names::LeftBound, "LeftBound"},
                                 {DiagonalBandMask_attributes::input_names::ShiftRightBound, "ShiftRightBound"},
                                 {DiagonalBandMask_attributes::input_names::B, "B"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(DiagonalBandMask_attributes::output_names,
                             {
                                 {DiagonalBandMask_attributes::output_names::Y, "Y"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Block_scale_quantize_attributes::input_names,
                             {
                                 {Block_scale_quantize_attributes::input_names::X, "X"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Block_scale_quantize_attributes::output_names,
                             {
                                 {Block_scale_quantize_attributes::output_names::Y, "Y"},
                                 {Block_scale_quantize_attributes::output_names::scale, "scale"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Block_scale_dequantize_attributes::input_names,
                             {
                                 {Block_scale_dequantize_attributes::input_names::X, "X"},
                                 {Block_scale_dequantize_attributes::input_names::scale, "scale"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Block_scale_dequantize_attributes::output_names,
                             {
                                 {Block_scale_dequantize_attributes::output_names::Y, "Y"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Concatenate_attributes::output_names,
                             {
                                 {Concatenate_attributes::output_names::Y, "Y"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Moe_grouped_matmul_attributes::input_names,
                             {
                                 {Moe_grouped_matmul_attributes::input_names::Token, "Token"},
                                 {Moe_grouped_matmul_attributes::input_names::Weight, "Weight"},
                                 {Moe_grouped_matmul_attributes::input_names::FirstTokenOffset, "FirstTokenOffset"},
                                 {Moe_grouped_matmul_attributes::input_names::TokenIndex, "TokenIndex"},
                                 {Moe_grouped_matmul_attributes::input_names::TokenKs, "TokenKs"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Moe_grouped_matmul_attributes::output_names,
                             {
                                 {Moe_grouped_matmul_attributes::output_names::Output, "Output"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Moe_grouped_matmul_bwd_attributes::input_names,
                             {
                                 {Moe_grouped_matmul_bwd_attributes::input_names::DOutput, "DOutput"},
                                 {Moe_grouped_matmul_bwd_attributes::input_names::Token, "Token"},
                                 {Moe_grouped_matmul_bwd_attributes::input_names::FirstTokenOffset, "FirstTokenOffset"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Moe_grouped_matmul_bwd_attributes::output_names,
                             {
                                 {Moe_grouped_matmul_bwd_attributes::output_names::DWeight, "DWeight"},
                             })

inline void
to_json(nlohmann::json& j, const Tensor_attributes& ta) {
    j = nlohmann::json{{"name", ta.name},
                       {"data_type", ta.data_type},
                       {"dim", ta.dim},
                       {"stride", ta.stride},
                       {"is_virtual", ta.is_virtual},
                       {"pass_by_value", ta.pass_by_value},
                       {"is_pass_by_value", ta.is_pass_by_value},
                       {"reordering_type", ta.reordering_type},
                       {"uid", ta.uid},
                       {"uid_assigned", ta.uid_assigned}};
    if (ta.ragged_offset) {
        j["ragged_offset_uid"]  = ta.ragged_offset->get_uid();
        j["ragged_offset_name"] = ta.ragged_offset->get_name();
    }
}

inline void
from_json(const nlohmann::json& j, Tensor_attributes& ta) {
    ta.name             = j.at("name").get<std::string>();
    ta.data_type        = j.at("data_type").get<DataType_t>();
    ta.dim              = j.at("dim").get<std::vector<int64_t>>();
    ta.stride           = j.at("stride").get<std::vector<int64_t>>();
    ta.is_virtual       = j.at("is_virtual").get<bool>();
    ta.is_pass_by_value = j.at("is_pass_by_value").get<bool>();
    ta.reordering_type  = j.at("reordering_type").get<TensorReordering_t>();
    ta.uid              = j.at("uid").get<Tensor_attributes::uid_t>();
    ta.uid_assigned     = j.at("uid_assigned").get<bool>();

    if (ta.is_pass_by_value && !j["pass_by_value"].is_null()) {
        ta.pass_by_value = j.at("pass_by_value");
    }
}

NLOHMANN_JSON_SERIALIZE_ENUM(KnobType_t,
                             {
                                 {KnobType_t::NOT_SET, nullptr},
                                 {KnobType_t::SWIZZLE, "SWIZZLE"},
                                 {KnobType_t::TILE_SIZE, "TILE_SIZE"},
                                 {KnobType_t::EDGE, "EDGE"},
                                 {KnobType_t::MULTIPLY, "MULTIPLY"},
                                 {KnobType_t::SPLIT_K_BUF, "SPLIT_K_BUF"},
                                 {KnobType_t::TILEK, "TILEK"},
                                 {KnobType_t::STAGES, "STAGES"},
                                 {KnobType_t::REDUCTION_MODE, "REDUCTION_MODE"},
                                 {KnobType_t::SPLIT_K_SLC, "SPLIT_K_SLC"},
                                 {KnobType_t::IDX_MODE, "IDX_MODE"},
                                 {KnobType_t::SPECFILT, "SPECFILT"},
                                 {KnobType_t::KERNEL_CFG, "KERNEL_CFG"},
                                 {KnobType_t::WORKSPACE, "WORKSPACE"},
                                 {KnobType_t::TILE_CGA_M, "TILE_CGA_M"},
                                 {KnobType_t::TILE_CGA_N, "TILE_CGA_N"},
                                 {KnobType_t::BLOCK_SIZE, "BLOCK_SIZE"},
                                 {KnobType_t::OCCUPANCY, "OCCUPANCY"},
                                 {KnobType_t::ARRAY_SIZE_PER_THREAD, "ARRAY_SIZE_PER_THREAD"},
                                 {KnobType_t::SPLIT_COLS, "SPLIT_COLS"},
                                 {KnobType_t::TILE_ROWS, "TILE_ROWS"},
                                 {KnobType_t::TILE_COLS, "TILE_COLS"},
                                 {KnobType_t::LOAD_SIZE, "LOAD_SIZE"},
                                 {KnobType_t::CTA_COUNT, "CTA_COUNT"},
                                 {KnobType_t::STREAM_K, "STREAM_K"},
                                 {KnobType_t::SPLIT_P_SLC, "SPLIT_P_SLC"},
                                 {KnobType_t::TILE_M, "TILE_M"},
                                 {KnobType_t::TILE_N, "TILE_N"},
                                 {KnobType_t::WARP_SPEC_CFG, "WARP_SPEC_CFG"},
                             })

#endif
}  // namespace cudnn_frontend::graph
