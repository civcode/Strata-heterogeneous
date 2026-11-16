#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace strata::program {
// The normal engine request summary, extracted from generate.cpp. Keep the
// log grammar stable for serve/server.py's ENGINE_REQUEST and echo_requests.
inline void log_request_summary(int64_t n, int64_t resume, int64_t read_n, int64_t fresh,
                                double prompt_ms, int64_t produced_n, double decode_ms,
                                int64_t draft_accepted, int64_t draft_offered,
                                std::size_t checkpoints, bool cancelled) {
    // "12288 of 98179" when cancelled mid-read (#471), the rate from what was read
    char read_txt[64];
    if (cancelled)
        std::snprintf(read_txt, sizeof(read_txt), "%lld of %lld", (long long) read_n, (long long) fresh);
    else
        std::snprintf(read_txt, sizeof(read_txt), "%lld", (long long) fresh);
    std::fprintf(stderr, "strata serve: prompt %lld tokens = %lld reused + %s read in %.0f ms (%.1f tok/s), "
                         "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld, %zu checkpoints%s\n",
                 (long long) n, (long long) resume, read_txt, prompt_ms,
                 prompt_ms > 0 ? 1000.0 * read_n / prompt_ms : 0.0, (long long) produced_n, decode_ms,
                 decode_ms > 0 ? 1000.0 * produced_n / decode_ms : 0.0, (long long) draft_accepted,
                 (long long) draft_offered, checkpoints, cancelled ? " (cancelled)" : "");
} 
} // namespace strata::program
