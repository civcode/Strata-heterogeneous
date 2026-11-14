#include "strata_heterogeneous/process.hpp"
#include "strata_heterogeneous/backend_domains.hpp"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

extern char** environ;

namespace strata::heterogeneous {
namespace {
constexpr int worker_fd = 3;

std::string error_string(int code) { return std::strerror(code); }

void close_fd(int& fd) noexcept {
    if (fd >= 0) { (void) ::close(fd); fd = -1; }
}

void reap(pid_t pid) noexcept {
    if (pid <= 0) return;
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
}

// CPU-only path boundary: validate all explicit per-stage paths before spawn.
void validate_stage_config_path(const std::string& path) {
    if (path.empty()) return; // legacy mock worker requires no --config
    if (path.size() < 2 || path.size() > 4096 || path.front() != '/' ||
        path.find('\0') != std::string::npos ||
        path.find('\n') != std::string::npos ||
        path.find('\r') != std::string::npos)
        throw std::invalid_argument("stage config path must be absolute and bounded");
    for (const auto& part : std::filesystem::path(path)) {
        if (part == "." || part == "..")
            throw std::invalid_argument("stage config path contains dot segment");
    }
}

struct SpawnActions {
    posix_spawn_file_actions_t actions{};
    SpawnActions() {
        const int rc = ::posix_spawn_file_actions_init(&actions);
        if (rc) throw std::runtime_error("spawn actions init: " + error_string(rc));
    }
    ~SpawnActions() { (void) ::posix_spawn_file_actions_destroy(&actions); }
    SpawnActions(const SpawnActions&) = delete;
    SpawnActions& operator=(const SpawnActions&) = delete;
    void close(int fd) {
        const int rc = ::posix_spawn_file_actions_addclose(&actions, fd);
        if (rc) throw std::runtime_error("spawn close action: " + error_string(rc));
    }
    void dup2(int from, int to) {
        const int rc = ::posix_spawn_file_actions_adddup2(&actions, from, to);
        if (rc) throw std::runtime_error("spawn dup2 action: " + error_string(rc));
    }
};

} // namespace

std::string plan_spec(const Plan& plan) {
    plan.validate();
    std::string spec;
    for (const auto& stage : plan.stages) {
        if (!spec.empty()) spec += ',';
        spec += stage.backend == Backend::cuda ? "cuda:" : "hip:";
        spec += std::to_string(stage.ordinal) + ':' + std::to_string(stage.first_layer)
             + '-' + std::to_string(stage.end_layer);
    }
    return spec;
}

std::string executable_directory() {
    char exe[4096]{};
    const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0 || static_cast<size_t>(n) >= sizeof(exe) - 1)
        throw std::runtime_error("cannot resolve current executable");
    std::string path(exe, static_cast<size_t>(n));
    const size_t slash = path.rfind('/');
    if (slash == std::string::npos) throw std::runtime_error("executable has no directory");
    return path.substr(0, slash);
}

WorkerProcessGroup::WorkerProcessGroup(WorkerProcessGroup&& other) noexcept
    : children_(std::move(other.children_)) { other.children_.clear(); }
WorkerProcessGroup& WorkerProcessGroup::operator=(WorkerProcessGroup&& other) noexcept {
    if (this != &other) {
        terminate();
        children_ = std::move(other.children_);
        other.children_.clear();
    }
    return *this;
}

WorkerProcessGroup WorkerProcessGroup::launch(const Plan& plan,
                                            const std::string& cuda_executable,
                                            const std::string& hip_executable, bool shared_memory,
                                            const std::string& cuda_config_path,
                                            const std::vector<std::string>& stage_config_paths, bool group_hip) {
    const auto domains = worker_domains(plan, group_hip);
    // An unsplit native reference uses one backend. Only the worker
    // executable(s) named by this plan must exist; mixed plans need both.
    for (const auto& stage : plan.stages) {
        const auto& path = stage.backend == Backend::cuda ? cuda_executable : hip_executable;
        if (path.empty() || path.front() != '/')
            throw std::invalid_argument("planned worker executable path must be absolute");
    }
    if (!cuda_config_path.empty() && (cuda_config_path.front() != '/' ||
        cuda_config_path.find('\n') != std::string::npos ||
        cuda_config_path.find('\r') != std::string::npos))
        throw std::invalid_argument("CUDA worker config path must be absolute and contain no line breaks");
    if (!stage_config_paths.empty()) {
        if (stage_config_paths.size() != plan.stages.size() ||
            !cuda_config_path.empty())
            throw std::invalid_argument("stage configs must match plan and cannot mix with legacy CUDA config");
        for (const auto& path : stage_config_paths) validate_stage_config_path(path);
    }
    if (group_hip) {
        if (stage_config_paths.size() != plan.stages.size())
            throw std::invalid_argument("grouped mode requires all physical stage configs");
        for (const auto& path : stage_config_paths)
            if (path.empty()) throw std::invalid_argument("grouped mode requires nonempty stage configs");
    }
    const std::string spec = plan_spec(plan);
    const std::string no_config;
    WorkerProcessGroup group;
    group.children_.reserve(domains.size()); // no allocation after a successful spawn
    for (const auto& domain : domains) {
        const size_t i = domain.terminal_stage();
        const std::string& exe = plan.stages[i].backend == Backend::cuda ? cuda_executable : hip_executable;
        int pair[2] = {-1, -1};
        if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, pair) < 0)
            throw std::runtime_error("worker socketpair: " + error_string(errno));
        try {
            // The socket source must not be fd 3: dup2(3,3) preserves CLOEXEC.
            // An earlier peer may have occupied fd 3, so close that end *first*.
            if (pair[1] == worker_fd) {
                const int moved = ::fcntl(pair[1], F_DUPFD_CLOEXEC, worker_fd + 1);
                if (moved < 0) throw std::runtime_error("worker socket duplication: " + error_string(errno));
                close_fd(pair[1]);
                pair[1] = moved;
            }
            SpawnActions actions;
            actions.close(pair[0]);
            actions.dup2(pair[1], worker_fd);
            actions.close(pair[1]);
            std::vector<std::string> args = {
                exe, shared_memory ? "--worker-shared" : "--worker", spec, std::to_string(plan.total_layers),
                std::to_string(plan.geometry.hc), std::to_string(plan.geometry.n_embd),
                std::to_string(plan.model_fingerprint), std::to_string(i)
            };
            const std::string& config_path = !stage_config_paths.empty()
                ? stage_config_paths[i]
                : (plan.stages[i].backend == Backend::cuda ? cuda_config_path : no_config);
            if (domain.size() > 1) {
                args[1] = shared_memory ? "--worker-hip-domain-shared" : "--worker-hip-domain";
                for (size_t j = domain.first_stage; j < domain.end_stage; ++j) {
                    args.push_back("--config");
                    args.push_back(stage_config_paths[j]);
                }
            } else if (!config_path.empty()) {
                args.push_back("--config");
                args.push_back(config_path);
            }
            std::vector<char*> argv;
            for (auto& a : args) argv.push_back(a.data());
            argv.push_back(nullptr);
            pid_t pid = -1;
            const int rc = ::posix_spawn(&pid, exe.c_str(), &actions.actions, nullptr, argv.data(), environ);
            if (rc) throw std::runtime_error("cannot launch worker: " + exe + ": " + error_string(rc));
            close_fd(pair[1]);
            group.children_.push_back({pair[0], pid});
            pair[0] = -1;
        } catch (...) {
            close_fd(pair[0]); close_fd(pair[1]);
            throw; // group destructor kills/reaps children already launched
        }
    }
    return group;
}

std::vector<int> WorkerProcessGroup::sockets() const {
    std::vector<int> fds;
    fds.reserve(children_.size());
    for (const auto& c : children_) fds.push_back(c.fd);
    return fds;
}

bool WorkerProcessGroup::finish(std::string& error, int grace_ms) {
    if (grace_ms < 0) throw std::invalid_argument("negative worker grace timeout");
    error.clear();
    for (auto& c : children_) close_fd(c.fd);
    bool ok = true;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(grace_ms);
    for (auto& c : children_) {
        if (c.pid <= 0) continue;
        int status = 0;
        pid_t result;
        while (true) {
            do { result = ::waitpid(c.pid, &status, WNOHANG); }
            while (result < 0 && errno == EINTR);
            if (result != 0 || std::chrono::steady_clock::now() >= deadline) break;
            ::usleep(10000);
        }
        if (result != c.pid) {
            if (result == 0) { (void) ::kill(c.pid, SIGKILL); reap(c.pid); }
            if (error.empty()) error = result == 0 ? "worker exceeded shutdown deadline" : "worker waitpid failed";
            ok = false;
        } else if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            if (error.empty()) error = "worker exited with failure";
            ok = false;
        }
        c.pid = -1;
    }
    children_.clear();
    return ok;
}

void WorkerProcessGroup::terminate() noexcept {
    // Do not block forever on a worker that ignores EOF or SIGTERM.
    for (auto& c : children_) close_fd(c.fd);
    for (auto& c : children_) if (c.pid > 0) (void) ::kill(c.pid, SIGKILL);
    for (auto& c : children_) { reap(c.pid); c.pid = -1; }
    children_.clear();
}

} // namespace strata::heterogeneous
