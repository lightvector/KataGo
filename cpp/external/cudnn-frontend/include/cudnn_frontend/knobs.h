#pragma once

namespace cudnn_frontend {

enum class KnobType_t {
    NOT_SET,

    SWIZZLE,
    TILE_SIZE,
    EDGE,
    MULTIPLY,
    SPLIT_K_BUF,
    TILEK,
    STAGES,
    REDUCTION_MODE,
    SPLIT_K_SLC,
    IDX_MODE,
    SPECFILT,
    KERNEL_CFG,
    WORKSPACE,
    TILE_CGA_M,
    TILE_CGA_N,
    BLOCK_SIZE,
    OCCUPANCY,
    ARRAY_SIZE_PER_THREAD,
    SPLIT_COLS,
    TILE_ROWS,
    TILE_COLS,
    LOAD_SIZE,
    CTA_COUNT,
    STREAM_K,
    SPLIT_P_SLC,
    TILE_M,
    TILE_N,
    WARP_SPEC_CFG,
};

class Knob {
   public:
    KnobType_t type  = KnobType_t::NOT_SET;
    int64_t maxValue = 0;
    int64_t minValue = 0;
    int64_t stride   = 0;

    Knob(KnobType_t type, int64_t max, int64_t min, int64_t str)
        : type(type), maxValue(max), minValue(min), stride(str) {}
};

static inline cudnnStatus_t
convert_to_backend_knob_type(KnobType_t const knob_type, cudnnBackendKnobType_t& cudnn_knob_type) {
    switch (knob_type) {
        case KnobType_t::SWIZZLE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_SWIZZLE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::TILE_SIZE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILE_SIZE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::EDGE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_EDGE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::MULTIPLY:
            cudnn_knob_type = CUDNN_KNOB_TYPE_MULTIPLY;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::SPLIT_K_BUF:
            cudnn_knob_type = CUDNN_KNOB_TYPE_SPLIT_K_BUF;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::TILEK:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILEK;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::STAGES:
            cudnn_knob_type = CUDNN_KNOB_TYPE_STAGES;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::REDUCTION_MODE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_REDUCTION_MODE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::SPLIT_K_SLC:
            cudnn_knob_type = CUDNN_KNOB_TYPE_SPLIT_K_SLC;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::IDX_MODE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_IDX_MODE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::SPECFILT:
            cudnn_knob_type = CUDNN_KNOB_TYPE_SPECFILT;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::KERNEL_CFG:
            cudnn_knob_type = CUDNN_KNOB_TYPE_KERNEL_CFG;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::WORKSPACE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_WORKSPACE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#if (CUDNN_VERSION >= 8600)
        case KnobType_t::TILE_CGA_M:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILE_CGA_M;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::TILE_CGA_N:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILE_CGA_N;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
#if (CUDNN_VERSION >= 8800)
        case KnobType_t::BLOCK_SIZE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_BLOCK_SIZE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
#if (CUDNN_VERSION >= 8900)
        case KnobType_t::OCCUPANCY:
            cudnn_knob_type = CUDNN_KNOB_TYPE_OCCUPANCY;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::ARRAY_SIZE_PER_THREAD:
            cudnn_knob_type = CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
#if (CUDNN_VERSION >= 8905)
        case KnobType_t::SPLIT_COLS:
            cudnn_knob_type = CUDNN_KNOB_TYPE_SPLIT_COLS;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::TILE_ROWS:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILE_ROWS;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::TILE_COLS:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILE_COLS;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::LOAD_SIZE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_LOAD_SIZE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
#if (CUDNN_VERSION >= 90700)
        case KnobType_t::CTA_COUNT:
            cudnn_knob_type = CUDNN_KNOB_TYPE_CTA_COUNT;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::STREAM_K:
            cudnn_knob_type = CUDNN_KNOB_TYPE_STREAM_K;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::SPLIT_P_SLC:
            cudnn_knob_type = CUDNN_KNOB_TYPE_SPLIT_P_SLC;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::TILE_M:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILE_M;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::TILE_N:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILE_N;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::WARP_SPEC_CFG:
            cudnn_knob_type = CUDNN_KNOB_TYPE_WARP_SPEC_CFG;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}

inline KnobType_t
convert_from_backend_knob_type(cudnnBackendKnobType_t cudnn_knob_type) {
    switch (cudnn_knob_type) {
        case CUDNN_KNOB_TYPE_SWIZZLE:
            return KnobType_t::SWIZZLE;
        case CUDNN_KNOB_TYPE_TILE_SIZE:
            return KnobType_t::TILE_SIZE;
        case CUDNN_KNOB_TYPE_EDGE:
            return KnobType_t::EDGE;
        case CUDNN_KNOB_TYPE_MULTIPLY:
            return KnobType_t::MULTIPLY;
        case CUDNN_KNOB_TYPE_SPLIT_K_BUF:
            return KnobType_t::SPLIT_K_BUF;
        case CUDNN_KNOB_TYPE_TILEK:
            return KnobType_t::TILEK;
        case CUDNN_KNOB_TYPE_STAGES:
            return KnobType_t::STAGES;
        case CUDNN_KNOB_TYPE_REDUCTION_MODE:
            return KnobType_t::REDUCTION_MODE;
        case CUDNN_KNOB_TYPE_SPLIT_K_SLC:
            return KnobType_t::SPLIT_K_SLC;
        case CUDNN_KNOB_TYPE_IDX_MODE:
            return KnobType_t::IDX_MODE;
        case CUDNN_KNOB_TYPE_SPECFILT:
            return KnobType_t::SPECFILT;
        case CUDNN_KNOB_TYPE_KERNEL_CFG:
            return KnobType_t::KERNEL_CFG;
        case CUDNN_KNOB_TYPE_WORKSPACE:
            return KnobType_t::WORKSPACE;
#if (CUDNN_VERSION >= 8600)
        case CUDNN_KNOB_TYPE_TILE_CGA_M:
            return KnobType_t::TILE_CGA_M;
        case CUDNN_KNOB_TYPE_TILE_CGA_N:
            return KnobType_t::TILE_CGA_N;
#endif
#if (CUDNN_VERSION >= 8800)
        case CUDNN_KNOB_TYPE_BLOCK_SIZE:
            return KnobType_t::BLOCK_SIZE;
#endif
#if (CUDNN_VERSION >= 8900)
        case CUDNN_KNOB_TYPE_OCCUPANCY:
            return KnobType_t::OCCUPANCY;
        case CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD:
            return KnobType_t::ARRAY_SIZE_PER_THREAD;
#endif
#if (CUDNN_VERSION >= 8905)
        case CUDNN_KNOB_TYPE_SPLIT_COLS:
            return KnobType_t::SPLIT_COLS;
        case CUDNN_KNOB_TYPE_TILE_ROWS:
            return KnobType_t::TILE_ROWS;
        case CUDNN_KNOB_TYPE_TILE_COLS:
            return KnobType_t::TILE_COLS;
        case CUDNN_KNOB_TYPE_LOAD_SIZE:
            return KnobType_t::LOAD_SIZE;
#endif
#if (CUDNN_VERSION >= 90700)
        case CUDNN_KNOB_TYPE_CTA_COUNT:
            return KnobType_t::CTA_COUNT;
        case CUDNN_KNOB_TYPE_STREAM_K:
            return KnobType_t::STREAM_K;
        case CUDNN_KNOB_TYPE_SPLIT_P_SLC:
            return KnobType_t::SPLIT_P_SLC;
        case CUDNN_KNOB_TYPE_TILE_M:
            return KnobType_t::TILE_M;
        case CUDNN_KNOB_TYPE_TILE_N:
            return KnobType_t::TILE_N;
        case CUDNN_KNOB_TYPE_WARP_SPEC_CFG:
            return KnobType_t::WARP_SPEC_CFG;
#endif
        default:
            return KnobType_t::NOT_SET;
    }
}

}  // namespace cudnn_frontend