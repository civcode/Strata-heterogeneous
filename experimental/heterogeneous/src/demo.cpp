// CPU-only subprocess smoke test: real exec-isolated worker binaries, no GPU.
#include "strata_heterogeneous/host.hpp"
#include "strata_heterogeneous/process.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace strata::heterogeneous;
namespace {
uint64_t uint_arg(const char* s) {
    if (!s || s[0] == '-') throw std::invalid_argument("negative number");
    size_t end = 0;
    const uint64_t x = std::stoull(s, &end, 0);
    if (end != std::strlen(s)) throw std::invalid_argument(std::string("invalid number: ") + s);
    return x;
}
uint32_t number(const char* s) {
    const auto x = uint_arg(s);
    if (x > UINT32_MAX) throw std::invalid_argument("32-bit number overflow");
    return static_cast<uint32_t>(x);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6 && argc != 7) {
            std::cerr << "usage: strata_heterogeneous_demo <plan> <layers> <hc> <n_embd> <model_fp> [rows]\n"
                         "  e.g.: strata_heterogeneous_demo cuda:0:0-3,hip:0:3-7,hip:1:7-10 10 2 6 12345 8\n";
            return 2;
        }
        const uint32_t layers = number(argv[2]);
        Geometry geometry{number(argv[3]), number(argv[4])};
        const uint64_t model_fp = uint_arg(argv[5]);
        const uint32_t rows = argc == 7 ? number(argv[6]) : 1;
        if (!rows) throw std::invalid_argument("rows must be nonzero");
        Plan plan = parse_plan(argv[1], layers, geometry, model_fp);
        const std::string dir = executable_directory();
        auto workers = WorkerProcessGroup::launch(plan,
            dir + "/strata_heterogeneous_mock_cuda_worker",
            dir + "/strata_heterogeneous_mock_hip_worker");
        const size_t worker_count = workers.size();
        Frame in;
        in.request_id = 1; in.window_id = 1;
        in.plan_fingerprint = plan.fingerprint(); in.geometry = plan.geometry; in.rows = rows;
        in.payload.assign(static_cast<size_t>(geometry.floats_per_row()) * rows, 0.0f);
        PipelineSession session(plan, workers.sockets());
        session.initialize();
        Frame actual = session.run_window(in);
        Frame expected = in;
        std::string err;
        if (!mock_execute(Stage{Backend::cuda, 0, 0, layers}, in, expected, err))
            throw std::runtime_error("reference mock execution failed: " + err);
        if (actual.payload != expected.payload)
            throw std::runtime_error("stage partition lost or reordered a layer");
        if (!workers.finish(err)) throw std::runtime_error("worker shutdown: " + err);
        std::cout << "PASS: " << worker_count << " exec-isolated CPU mock workers, " << rows
                  << " rows, " << layers << " contiguous layers, " << actual.payload.size()
                  << " float32 handoff values, final checksum/parity verified (no GPU)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "heterogeneous demo: " << e.what() << '\n';
        return 1;
    }
}
