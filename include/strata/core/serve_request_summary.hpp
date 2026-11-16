#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace strata::core {

struct ServeRequestSummary {
    int64_t prompt_tokens = 0, reused = 0, prompt_read = 0, generated = 0;
    double prompt_ms = 0, decode_ms = 0;
    int64_t drafts_accepted = 0, drafts_offered = 0;
    std::size_t checkpoints = 0;
    bool cancelled = false;
};

// This is the public request-summary line, also consumed by serve/server.py's
// ENGINE_REQUEST parser and STRATA_REQUEST_LINES log follower.
inline void print_serve_request_summary(std::FILE* out, const ServeRequestSummary& s) {
    const int64_t fresh = s.prompt_tokens - s.reused;
    char read_txt[64];
    if (s.cancelled)
        std::snprintf(read_txt, sizeof(read_txt), "%lld of %lld", (long long) s.prompt_read, (long long) fresh);
    else
        std::snprintf(read_txt, sizeof(read_txt), "%lld", (long long) fresh);
    std::fprintf(out, "strata serve: prompt %lld tokens = %lld reused + %s read in %.0f ms (%.1f tok/s), "
                      "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld, %zu checkpoints%s\n",
                 (long long) s.prompt_tokens, (long long) s.reused, read_txt, s.prompt_ms,
                 s.prompt_ms > 0 ? 1000.0 * s.prompt_read / s.prompt_ms : 0.0, (long long) s.generated, s.decode_ms,
                 s.decode_ms > 0 ? 1000.0 * s.generated / s.decode_ms : 0.0, (long long) s.drafts_accepted,
                 (long long) s.drafts_offered, s.checkpoints, s.cancelled ? " (cancelled)" : "");
}

} // namespace strata::core
