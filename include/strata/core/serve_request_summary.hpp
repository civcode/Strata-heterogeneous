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

// Shared application wire records. The caller controls flushing, preserving
// batch emission and the optional vanilla DONE telemetry suffix.
inline void print_serve_ready(std::FILE* out, int64_t context) {
    std::fprintf(out, "READY %lld stop\n", (long long)context);
}
inline void print_serve_reused(std::FILE* out, int64_t reused) {
    std::fprintf(out, "REUSED %lld\n", (long long)reused);
}
inline void print_serve_token(std::FILE* out, int32_t token) {
    std::fprintf(out, "T %d\n", (int)token);
}
inline void print_serve_progress(std::FILE* out, int64_t done, int64_t total, int64_t from, double ms) {
    std::fprintf(out, "PP %lld %lld %.0f %.1f\n", (long long)done, (long long)total, ms,
                 ms > 0.0 ? 1000.0 * (double)(done - from) / ms : 0.0);
}
inline void print_serve_done_prefix(std::FILE* out, const ServeRequestSummary& s, const char* finish) {
    std::fprintf(out, "DONE %lld %lld %.1f %.1f %s %lld %lld %lld",
                 (long long)s.generated, (long long)s.prompt_tokens, s.prompt_ms, s.decode_ms, finish,
                 (long long)s.drafts_accepted, (long long)s.drafts_offered, (long long)s.reused);
}

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
