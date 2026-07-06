// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"

// keep sync with BlockAttentionBiasEnum
enum class bias_enum
{
    no_bias          = 0,
    elementwise_bias = 1,
    alibi            = 2,
};

struct bias_info
{
    bias_enum type;
    /*
     * simple dispatch logic
     *
     * if type == elementwise_bias:
     *      if rank_info == 0:
     *           bias is 1*1*s*s
     *      elif rank_info == 1:
     *           bias is 1*h*s*s
     *      elif rank_info == 2:
     *           bias is b*h*s*s
     *
     * elif type == alibi:
     *       if rank_info == 0:
     *           alibi in 1*h
     *       elif rank_info == 1:
     *           alibi in b*h
     */
    int rank_info;

    void serialize(std::ostream& os) const
    {
        if(type == bias_enum::no_bias)
            os << "n";
        else if(type == bias_enum::elementwise_bias)
        {
            os << "e";
            if(rank_info != 0)
            {
                os << "[" << rank_info << "]";
            }
        }
        else if(type == bias_enum::alibi)
        {
            os << "alibi";
            if(rank_info != 0)
            {
                os << "[" << rank_info << "]";
            }
        }
    }

    static bias_info decode(std::string str)
    {
        bias_info info{bias_enum::no_bias, 0};
        auto found_0 = str.find(':');
        if(found_0 != std::string::npos)
        {
            std::string t = str.substr(0, found_0);
            std::string v = str.substr(found_0 + 1);
            if(t == "e" || t == "elementwise")
            {
                info.type      = bias_enum::elementwise_bias;
                info.rank_info = std::stoi(v);
                if(info.rank_info < 0 || info.rank_info > 2)
                    throw std::invalid_argument("invalid bias rank: " + str);
            }
            else if(t == "a" || t == "alibi")
            {
                info.type      = bias_enum::alibi;
                info.rank_info = std::stoi(v);
                if(info.rank_info < 0 || info.rank_info > 1)
                    throw std::invalid_argument("invalid bias rank: " + str);
            }
            else
            {
                throw std::invalid_argument("invalid bias value: " + str);
            }
        }
        else if(str == "0" || str == "n")
        {
            info.type = bias_enum::no_bias;
        }
        else if(str == "1" || str == "e" || str == "elementwise")
        {
            info.type = bias_enum::elementwise_bias;
        }
        else if(str == "2" || str == "a" || str == "alibi")
        {
            info.type = bias_enum::alibi;
        }
        else
        {
            throw std::invalid_argument("invalid bias value: " + str);
        }
        return info;
    }

    friend std::ostream& operator<<([[clang::lifetimebound]] std::ostream& os, const bias_info& bi)
    {
        bi.serialize(os);
        return os;
    }
};
