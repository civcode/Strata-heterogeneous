// Phase 11B TEST-ONLY HIP startup fault. Not linked into the production binary.
// Load with LD_PRELOAD only in a bounded, monitored, exclusive GPU validation.
// Fail precisely one hipStreamCreateWithFlags on a chosen *logical* HIP device,
// after N matching calls of that symbol on that same device.
#include <hip/hip_runtime_api.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {
std::atomic<unsigned> targeted_calls{0};
std::atomic<bool> fired{false};

bool number(const char* name, unsigned& n) {
    const char* s = std::getenv(name);
    if (!s || !*s) return false;
    char* end = nullptr;
    const unsigned long v = std::strtoul(s, &end, 10);
    if (s == end || *end || v > 100000UL) return false;
    n = static_cast<unsigned>(v);
    return true;
}

using Create = hipError_t (*)(hipStream_t*, unsigned int);
Create next_create() {
    static Create real = reinterpret_cast<Create>(dlsym(RTLD_NEXT, "hipStreamCreateWithFlags"));
    return real;
}
} // namespace

extern "C" hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
    // Without all three explicit test-only settings, behave exactly as the runtime.
    Create real = next_create();
    if (!real) {
        std::fputs("PHASE11B_FAULT_FATAL real hipStreamCreateWithFlags not found\n", stderr);
        std::fflush(stderr);
        std::_Exit(126);
    }
    if (const char* enabled = std::getenv("STRATA_PHASE11B_FAULT_ENABLE");
        enabled && std::strcmp(enabled, "stream-create-once") == 0) {
        unsigned target = 0, skip = 0;
        if (number("STRATA_PHASE11B_FAULT_DEVICE", target) &&
            number("STRATA_PHASE11B_FAULT_SKIP", skip)) {
            int current = -1;
            if (hipGetDevice(&current) == hipSuccess && current >= 0 &&
                static_cast<unsigned>(current) == target) {
                const unsigned at = targeted_calls.fetch_add(1, std::memory_order_relaxed);
                if (at == skip && !fired.exchange(true, std::memory_order_acq_rel)) {
                    std::fprintf(stderr,
                        "PHASE11B_FAULT_INJECTED stream-create device=%d skipped=%u\n",
                        current, skip);
                    std::fflush(stderr);
                    return hipErrorOutOfMemory;
                }
            }
        }
    }
    return real(stream, flags);
}
