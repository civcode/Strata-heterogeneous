#include "strata_heterogeneous/hip_worker_bootstrap.hpp"
#include "strata_heterogeneous/backend_domains.hpp"
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::heterogeneous {
namespace {
uint64_t decimal(const char* text) {
    if (!text) throw std::invalid_argument("missing HIP worker number");
    const std::string s(text);
    if (s.empty() || s.front() == '-' || s.front() == '+')
        throw std::invalid_argument("invalid HIP worker number");
    uint64_t n = 0;
    const auto result = std::from_chars(s.data(), s.data() + s.size(), n);
    if (result.ec != std::errc{} || result.ptr != s.data() + s.size())
        throw std::invalid_argument("invalid HIP worker number");
    return n;
}
uint32_t u32(const char* s) {
    const uint64_t n = decimal(s);
    if (n > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("HIP worker number exceeds u32");
    return static_cast<uint32_t>(n);
}
void check_path(const std::string& path) {
    if (path.size() < 2 || path.size() > 4096 || path.front() != '/' ||
        path.find('\0') != std::string::npos ||
        path.find('\n') != std::string::npos ||
        path.find('\r') != std::string::npos)
        throw std::invalid_argument("HIP worker config path must be absolute and bounded");
    for (const auto& part : std::filesystem::path(path)) {
        if (part == "." || part == "..")
            throw std::invalid_argument("HIP worker config path contains dot segment");
    }
}
} // namespace

bool parse_hip_worker_invocation(int argc, const char* const* argv,
                                 HipWorkerInvocation& out, std::string& error) {
    try {
        if ((argc != 10 && argc != 12) || !argv)
            throw std::invalid_argument("expected HIP worker mode plan layers hc n_embd fingerprint index --config /absolute/path");
        for (int i = 0; i < argc; ++i)
            if (!argv[i]) throw std::invalid_argument("missing HIP worker argument");
        const std::string mode(argv[1]);
        const bool grouped = mode == "--worker-hip-domain" || mode == "--worker-hip-domain-shared";
        if ((!grouped && mode != "--worker" && mode != "--worker-shared") ||
            argc != (grouped ? 12 : 10) || std::string(argv[8]) != "--config")
            throw std::invalid_argument("HIP worker mode/config argument mismatch");
        HipWorkerInvocation next;
        next.shared_memory = mode == "--worker-shared" || mode == "--worker-hip-domain-shared";
        next.plan = parse_plan(argv[2], u32(argv[3]),
                               {u32(argv[4]), u32(argv[5])}, decimal(argv[6]));
        next.stage_index = u32(argv[7]);
        if (next.stage_index >= next.plan.stages.size() ||
            next.plan.stages[next.stage_index].backend != Backend::hip)
            throw std::invalid_argument("HIP worker stage index/backend mismatch");
        next.config_path = argv[9];
        check_path(next.config_path);
        if (grouped) {
            const auto domains = worker_domains(next.plan, true);
            if (next.stage_index != domains.back().terminal_stage() ||
                std::string(argv[10]) != "--config")
                throw std::invalid_argument("HIP domain requires terminal stage and both configs");
            check_path(argv[11]);
            next.domain_configs = {argv[9], argv[11]};
            if (next.domain_configs[0] == next.domain_configs[1])
                throw std::invalid_argument("HIP domain configs must be distinct");
        }
        out = std::move(next);
        error.clear();
        return true;
    } catch (const std::exception& ex) {
        error = ex.what();
        return false;
    }
}
} // namespace strata::heterogeneous
