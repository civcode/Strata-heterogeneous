#pragma once
#include <cstdio>
#include <cstdlib>

namespace strata::core {
// GPU work may still reference caller-owned storage. Do not unwind automatic owners or run
// static destructors. This is the same process-terminal policy used by the generation boundary.
[[noreturn]] inline void transfer_terminal(const char* context, const char* error) {
    std::fprintf(stderr, "strata: %s: %s; exiting without releasing transfer owners\n", context, error);
    std::printf("ERR %s: %s\n", context, error);
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(1);
}
} // namespace strata::core
