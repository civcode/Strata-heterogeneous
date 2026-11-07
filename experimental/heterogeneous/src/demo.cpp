// A CPU-only subprocess smoke test for an eventual CUDA/HIP stage coordinator.
// All workers use mock_execute. No GPU runtime is linked into this binary.
#include "strata_heterogeneous/host.hpp"

#include <cerrno>
#include <fcntl.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;
using namespace strata::heterogeneous;
namespace {
constexpr int worker_fd = 3;
struct Child { int fd; pid_t pid; };
uint64_t uint_arg(const char* s) {
    size_t end = 0;
    const uint64_t x = std::stoull(s, &end, 0);
    if (end != std::strlen(s)) throw std::invalid_argument(std::string("invalid number: ") + s);
    return x;
}
std::string executable_path() {
    char p[4096]{};
    const ssize_t n = ::readlink("/proc/self/exe", p, sizeof(p) - 1);
    if (n <= 0) throw std::runtime_error("cannot resolve current executable");
    return std::string(p, static_cast<size_t>(n));
}
Child spawn_mock(const std::string& exe, const std::vector<std::string>& common, size_t stage_index) {
    int fds[2]{};
    if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) != 0)
        throw std::runtime_error("socketpair failed");
    // If fd 3 is the child socket itself, dup2(3,3) would not clear its
    // CLOEXEC bit. Move the source to a different descriptor first.
    if (fds[1] == worker_fd) {
        const int moved = ::fcntl(fds[1], F_DUPFD_CLOEXEC, worker_fd + 1);
        if (moved < 0) throw std::runtime_error("socket fd duplication failed");
        ::close(fds[1]);
        fds[1] = moved;
    }
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) throw std::runtime_error("spawn actions init failed");
    // Close the parent's end FIRST. It can be fd 3, which dup2 needs later.
    posix_spawn_file_actions_addclose(&actions, fds[0]);
    posix_spawn_file_actions_adddup2(&actions, fds[1], worker_fd);
    posix_spawn_file_actions_addclose(&actions, fds[1]);
    std::vector<std::string> args = {exe, "--worker"};
    args.insert(args.end(), common.begin(), common.end());
    args.push_back(std::to_string(stage_index));
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    pid_t pid = -1;
    const int rc = posix_spawn(&pid, exe.c_str(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    ::close(fds[1]);
    if (rc != 0) { ::close(fds[0]); throw std::runtime_error(std::string("worker spawn failed: ") + std::strerror(rc)); }
    return {fds[0], pid};
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 6 || argc > 8) {
            std::cerr << "usage: strata_heterogeneous_demo <plan> <layers> <hc> <n_embd> <model_fp> [rows]\n"
                      << "  e.g.: strata_heterogeneous_demo cuda:0:0-3,hip:0:3-7,hip:1:7-10 10 2 6 12345 8\n";
            return 2;
        }
        const bool child = std::string(argv[1]) == "--worker";
        const int offset = child ? 2 : 1;
        if (argc != (child ? 8 : 6) && argc != (child ? 8 : 7))
            throw std::invalid_argument("incorrect number of args");
        const auto number = [](const char* s) -> uint32_t {
            auto v = uint_arg(s);
            if (v > UINT32_MAX) throw std::invalid_argument("32-bit number overflow");
            return static_cast<uint32_t>(v);
        };
        const uint32_t layers = number(argv[offset + 1]);
        Geometry geometry{number(argv[offset + 2]), number(argv[offset + 3])};
        const uint64_t model_fp = uint_arg(argv[offset + 4]);
        Plan plan = parse_plan(argv[offset], layers, geometry, model_fp);
        if (child) {
            const size_t index = uint_arg(argv[offset + 5]);
            serve_worker(worker_fd, plan, index, mock_execute);
            return 0;
        }
        const uint32_t rows = argc == 7 ? number(argv[6]) : 1;
        if (!rows) throw std::invalid_argument("rows must be nonzero");
        std::vector<std::string> common = {argv[1], argv[2], argv[3], argv[4], argv[5]};
        const std::string exe = executable_path();
        std::vector<Child> children;
        std::vector<int> fds;
        for (size_t i = 0; i < plan.stages.size(); ++i) {
            auto w = spawn_mock(exe, common, i);
            children.push_back(w); fds.push_back(w.fd);
        }
        Frame in;
        in.request_id = 1; in.window_id = 1;
        in.plan_fingerprint = plan.fingerprint(); in.geometry = plan.geometry; in.rows = rows;
        in.payload.assign(static_cast<size_t>(geometry.floats_per_row()) * rows, 0.0f);
        Frame actual = run_pipeline(plan, fds, in);
        Frame expected = in;
        std::string ignored;
        mock_execute(Stage{Backend::cuda, 0, 0, layers}, in, expected, ignored);
        for (auto& w : children) ::close(w.fd);
        for (const auto& w : children) {
            int status = 0;
            if (::waitpid(w.pid, &status, 0) != w.pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
                throw std::runtime_error("worker terminated abnormally");
        }
        if (actual.payload != expected.payload) throw std::runtime_error("stage partition lost or reordered a layer");
        std::cout << "PASS: " << children.size() << " isolated worker processes, " << rows
                  << " rows, " << layers << " contiguous layers, " << actual.payload.size()
                  << " float32 handoff values, final checksum/parity verified (CPU mock; no GPU)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "heterogeneous demo: " << e.what() << '\n';
        return 1;
    }
}
