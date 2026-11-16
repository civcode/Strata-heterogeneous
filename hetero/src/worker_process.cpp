#include "strata/hetero/worker_process.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>

extern char** environ;

namespace strata::hetero {

WorkerProcess::WorkerProcess(WorkerProcess&& o) noexcept
    : fd_(o.fd_), pid_(o.pid_), drain_ms_(o.drain_ms_), reaped_(o.reaped_), status_(o.status_) {
    o.fd_ = -1; o.pid_ = -1; o.reaped_ = false;
}

WorkerProcess& WorkerProcess::operator=(WorkerProcess&& o) noexcept {
    if (this != &o) {
        if (pid_ > 0) {
            std::string err;
            if (!finish(err) && !reaped_) std::fprintf(stderr, "strata hetero: %s\n", err.c_str());
        }
        close_socket();
        fd_ = o.fd_; pid_ = o.pid_; drain_ms_ = o.drain_ms_; reaped_ = o.reaped_; status_ = o.status_;
        o.fd_ = -1; o.pid_ = -1; o.reaped_ = false;
    }
    return *this;
}

WorkerProcess::~WorkerProcess() noexcept {
    if (pid_ <= 0) { close_socket(); return; }
    std::string err;
    if (!finish(err) && !reaped_)
        std::fprintf(stderr, "strata hetero: %s\n", err.c_str());   // still running: named, never signalled
    pid_ = -1;
}

void WorkerProcess::close_socket() noexcept {
    if (fd_ >= 0) (void) ::close(fd_);
    fd_ = -1;
}

bool WorkerProcess::launch(const std::string& exe, const std::vector<std::string>& args, int drain_ms,
                           WorkerProcess& out, std::string& err) {
    if (out.started()) { err = "worker: already started"; return false; }
    if (exe.empty() || exe.front() != '/') { err = "worker: the executable path must be absolute"; return false; }
    if (drain_ms < 0) { err = "worker: negative drain time"; return false; }
    int pair[2] = {-1, -1};
    if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, pair) != 0) {
        err = std::string("worker: socketpair: ") + std::strerror(errno);
        return false;
    }
    auto fail = [&](const std::string& why) {
        if (pair[0] >= 0) (void) ::close(pair[0]);
        if (pair[1] >= 0) (void) ::close(pair[1]);
        err = why;
        return false;
    };
    // The child's end must not already be descriptor 3: dup2(3, 3) would keep its close-on-exec flag.
    if (pair[1] == kWorkerSocketFd) {
        const int moved = ::fcntl(pair[1], F_DUPFD_CLOEXEC, kWorkerSocketFd + 1);
        if (moved < 0) return fail(std::string("worker: fcntl: ") + std::strerror(errno));
        (void) ::close(pair[1]);
        pair[1] = moved;
    }
    posix_spawn_file_actions_t fa;
    if (::posix_spawn_file_actions_init(&fa) != 0) return fail("worker: spawn actions");
    int rc = ::posix_spawn_file_actions_adddup2(&fa, pair[1], kWorkerSocketFd);
    std::vector<std::string> store;
    store.push_back(exe);
    for (const auto& a : args) store.push_back(a);
    std::vector<char*> argv;
    for (auto& a : store) argv.push_back(a.data());
    argv.push_back(nullptr);
    pid_t pid = -1;
    if (rc == 0) rc = ::posix_spawn(&pid, exe.c_str(), &fa, nullptr, argv.data(), environ);
    (void) ::posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) return fail("worker: cannot start " + exe + ": " + std::strerror(rc));
    (void) ::close(pair[1]);
    out.fd_ = pair[0];
    out.pid_ = pid;
    out.drain_ms_ = drain_ms;
    out.reaped_ = false;
    out.status_ = 0;
    return true;
}

bool WorkerProcess::ended(int* exit_code) {
    if (pid_ <= 0) return true;
    if (!reaped_) {
        int st = 0;
        pid_t r;
        do { r = ::waitpid(pid_, &st, WNOHANG); } while (r < 0 && errno == EINTR);
        if (r == pid_) { reaped_ = true; status_ = st; }
        else if (r < 0) { reaped_ = true; status_ = -1; }   // not our child any more: nothing left to wait for
    }
    if (reaped_ && exit_code != nullptr) *exit_code = (status_ != -1 && WIFEXITED(status_)) ? WEXITSTATUS(status_) : -1;
    return reaped_;
}

bool WorkerProcess::finish(std::string& err) {
    err.clear();
    close_socket();   // the worker's serve loop ends on this
    if (pid_ <= 0) return true;
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(drain_ms_);
    int code = -1;
    while (!ended(&code)) {
        if (std::chrono::steady_clock::now() >= until) {
            err = "the front stage worker is still running after " + std::to_string(drain_ms_) +
                  " ms; it was not signalled, PID " + std::to_string((long) pid_);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (code != 0) {
        err = code < 0 ? "the front stage worker ended abnormally"
                       : "the front stage worker ended with exit code " + std::to_string(code);
        return false;
    }
    return true;
}

} // namespace strata::hetero
