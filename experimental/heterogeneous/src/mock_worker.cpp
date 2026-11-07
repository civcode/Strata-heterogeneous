// Backend-tagged CPU test worker.  These two executables intentionally link
// no CUDA or HIP runtime.  Their identity check proves process isolation
// and launch wiring, not availability of the claimed GPU hardware.
#include "strata_heterogeneous/host.hpp"

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>

#ifndef STRATA_HETERO_MOCK_BACKEND
#error "Compile a mock worker with an explicit backend marker"
#endif

using namespace strata::heterogeneous;
namespace {
uint64_t parse_unsigned(const char* str) {
    std::string s(str);
    if (s.empty() || s[0] == '-') throw std::invalid_argument("bad unsigned integer");
    size_t used = 0;
    const auto value = std::stoull(s, &used, 10);
    if (used != s.size()) throw std::invalid_argument("bad unsigned integer");
    return value;
}
uint32_t parse_u32(const char* str) {
    const auto n = parse_unsigned(str);
    if (n > UINT32_MAX) throw std::invalid_argument("unsigned integer exceeds u32");
    return static_cast<uint32_t>(n);
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 8 || std::string(argv[1]) != "--worker")
            throw std::invalid_argument("expected --worker plan layers hc n_embd fingerprint stage_index");
        Plan plan = parse_plan(argv[2], parse_u32(argv[3]),
                               {parse_u32(argv[4]), parse_u32(argv[5])}, parse_unsigned(argv[6]));
        const auto index = parse_u32(argv[7]);
        if (index >= plan.stages.size() ||
            static_cast<unsigned>(plan.stages[index].backend) != STRATA_HETERO_MOCK_BACKEND)
            throw std::runtime_error("mock backend does not match configured stage");
        // The fd is supplied only by the parent through posix_spawn dup2.
        int type = 0;
        socklen_t len = sizeof(type);
        if (::getsockopt(3, SOL_SOCKET, SO_TYPE, &type, &len) != 0 || type != SOCK_STREAM)
            throw std::runtime_error("no worker socket on descriptor 3");
        MockStageExecutor executor;
        serve_worker_stateful(3, plan, index, executor);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "mock worker: " << ex.what() << '\n';
        return 1;
    }
}
