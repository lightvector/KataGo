

/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>

namespace cudnn_frontend {

static const char *
get_environment(const char *name) {
#ifdef WIN32
#pragma warning(disable : 4996)
#define _CRT_SECURE_NO_WARNINGS
#endif

    return std::getenv(name);
}

inline int
getLogLevel() {
#ifdef NV_CUDNN_FRONTEND_DISABLE_LOGGING
    static int log_level = 0;
#else
    static int log_level = []() {
        const char *env_val = get_environment("CUDNN_FRONTEND_LOG_INFO");
        return env_val ? std::atoi(env_val) : 0;
    }();
#endif
    return log_level;
}

inline bool &
isLoggingEnabled() {
#ifdef NV_CUDNN_FRONTEND_DISABLE_LOGGING
    static bool log_enabled = false;
#else
    static bool log_enabled = (getLogLevel() > 0);
#endif
    return log_enabled;
}

inline bool &
isLoggingTensorDumpEnabled() {
#ifdef NV_CUDNN_FRONTEND_DISABLE_LOGGING
    static bool tensor_dump_enabled = false;
#else
    static bool tensor_dump_enabled = []() {
        int level = getLogLevel();
        return level >= 1 && level < 10;
    }();
#endif
    return tensor_dump_enabled;
}

inline std::ostream &
getStream() {
    static std::ofstream outFile;
    static std::ostream &stream =
        get_environment("CUDNN_FRONTEND_LOG_FILE")
            ? (std::strncmp(get_environment("CUDNN_FRONTEND_LOG_FILE"), "stdout", 6) == 0
                   ? std::cout
                   : (std::strncmp(get_environment("CUDNN_FRONTEND_LOG_FILE"), "stderr", 6) == 0
                          ? std::cerr
                          : (outFile.open(get_environment("CUDNN_FRONTEND_LOG_FILE"), std::ios::out), outFile)))
            : (isLoggingEnabled() = false, std::cout);
    return stream;
}

class ConditionalStreamer {
   private:
    std::ostream &stream;

   public:
    ConditionalStreamer(std::ostream &stream_) : stream(stream_) {}

    template <typename T>
    const ConditionalStreamer &
    operator<<(const T &t) const {
        if (isLoggingEnabled()) {
            stream << t;
        }
        return *this;
    }

    const ConditionalStreamer &
    operator<<(std::ostream &(*spl)(std::ostream &)) const {
        if (isLoggingEnabled()) {
            stream << spl;
        }
        return *this;
    }
};

inline ConditionalStreamer &
getLogger() {
    static ConditionalStreamer opt(getStream());
    return opt;
}

#define CUDNN_FE_LOG(X)           \
    do {                          \
        if (isLoggingEnabled()) { \
            getLogger() << X;     \
        }                         \
    } while (0);

#define CUDNN_FE_LOG_LABEL(X)                        \
    do {                                             \
        if (isLoggingEnabled()) {                    \
            getLogger() << "[cudnn_frontend] " << X; \
        }                                            \
    } while (0);

#define CUDNN_FE_LOG_LABEL_ENDL(X)                                \
    do {                                                          \
        if (isLoggingEnabled()) {                                 \
            getLogger() << "[cudnn_frontend] " << X << std::endl; \
        }                                                         \
    } while (0);

#define CUDNN_FE_LOG_BANNER(X)                                                         \
    do {                                                                               \
        if (isLoggingEnabled()) {                                                      \
            {                                                                          \
                constexpr int total_width = 128;                                       \
                std::ostringstream oss;                                                \
                oss << "[cudnn_frontend] ||| === " << X << " === |||";                 \
                std::string banner_line = oss.str();                                   \
                int banner_len          = static_cast<int>(banner_line.size());        \
                int pad                 = total_width - banner_len;                    \
                if (pad > 0) {                                                         \
                    banner_line.insert(banner_line.size() - 5, std::string(pad, ' ')); \
                }                                                                      \
                getLogger() << std::string(total_width, '=') << std::endl;             \
                getLogger() << banner_line << std::endl;                               \
                getLogger() << std::string(total_width, '=') << std::endl;             \
            }                                                                          \
        }                                                                              \
    } while (0);

static std::ostream &
operator<<(std::ostream &os, const BackendDescriptor &desc) {
    if (isLoggingEnabled()) {
        os << desc.describe();
    }
    return os;
}

}  // namespace cudnn_frontend
