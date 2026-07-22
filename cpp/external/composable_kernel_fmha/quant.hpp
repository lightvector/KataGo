// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

// keep sync with BlockAttentionQuantScaleEnum
enum class quant_scale_enum
{
    no_scale      = 0,
    pertensor     = 1,
    blockscale    = 2,
    kv_blockscale = 3, // Q per-tensor, K/V per-page block scale
    mx            = 4, // Microscaling (MX)
};

struct quant_scale_info
{
    quant_scale_enum type;

    void serialize(std::ostream& os) const
    {
        if(type == quant_scale_enum::no_scale)
            os << "n";
        else if(type == quant_scale_enum::pertensor)
            os << "pt";
        else if(type == quant_scale_enum::blockscale)
            os << "bs";
        else if(type == quant_scale_enum::kv_blockscale)
            os << "kvbs";
        else if(type == quant_scale_enum::mx)
            os << "mx";
    }

    static quant_scale_info decode(std::string str)
    {
        quant_scale_info info{quant_scale_enum::no_scale};
        if(str == "n" || str == "0")
        {
            info.type = quant_scale_enum::no_scale;
        }
        else if(str == "pt" || str == "1")
        {
            info.type = quant_scale_enum::pertensor;
        }
        else if(str == "bs" || str == "2")
        {
            info.type = quant_scale_enum::blockscale;
        }
        else if(str == "kvbs" || str == "3")
        {
            info.type = quant_scale_enum::kv_blockscale;
        }
        else if(str == "mx" || str == "4")
        {
            info.type = quant_scale_enum::mx;
        }
        else
        {
            throw std::invalid_argument("invalid quant scale value: " + str);
        }
        return info;
    }

    friend std::ostream& operator<<(std::ostream& os, const quant_scale_info& qsi)
    {
        qsi.serialize(os);
        return os;
    }
};
#pragma clang diagnostic pop
