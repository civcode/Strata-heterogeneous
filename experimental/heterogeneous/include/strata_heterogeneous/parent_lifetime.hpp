#pragma once

#include <charconv>
#include <csignal>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/prctl.h>
#include <sys/types.h>
#include <unistd.h>

namespace strata::heterogeneous {

// Arm the same Linux parent-death guarantee as the vanilla Python launcher.
// Check both sides of prctl: the coordinator may exit before this worker execs.
inline void bind_worker_lifetime(const std::string& pid_text) {
    unsigned long pid = 0;
    auto parsed = std::from_chars(pid_text.data(), pid_text.data()+pid_text.size(), pid);
    if (parsed.ec != std::errc{} || parsed.ptr != pid_text.data()+pid_text.size() ||
        pid <= 1 || pid > static_cast<unsigned long>(std::numeric_limits<pid_t>::max()))
        throw std::invalid_argument("invalid coordinator PID");
    const auto parent = static_cast<pid_t>(pid);
    if (::getppid() != parent || ::prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0) != 0 ||
        ::getppid() != parent)
        throw std::runtime_error("cannot bind worker lifetime to coordinator");
}

// Keep the existing diagnostic bootstrap parsers unchanged. Older worker
// binaries reject this new trailing capability instead of silently ignoring it.
inline void consume_worker_lifetime_arg(int& argc, char** argv) {
    if (argc >= 3 && std::string(argv[argc-2]) == "--coordinator-pid") {
        bind_worker_lifetime(argv[argc-1]);
        argc -= 2;
        argv[argc] = nullptr;
    }
}

} // namespace strata::heterogeneous
