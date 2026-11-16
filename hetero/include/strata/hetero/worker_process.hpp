#pragma once

// The front stage worker as a child process of the host.
//
// The worker is an executable built for the other GPU runtime.  It inherits one AF_UNIX socket on descriptor 3
// (kWorkerSocketFd) and serves the front link on it.  The host never signals the worker: a GPU process that is
// killed can leave its card in a bad state.  Closing the socket asks it to end; the host then waits for it.
//
// Host-only code: nothing here touches CUDA or HIP.

#include <string>
#include <sys/types.h>
#include <vector>

namespace strata::hetero {

constexpr int kWorkerSocketFd = 3;

class WorkerProcess {
public:
    WorkerProcess() = default;
    ~WorkerProcess() noexcept;
    WorkerProcess(const WorkerProcess&) = delete;
    WorkerProcess& operator=(const WorkerProcess&) = delete;
    WorkerProcess(WorkerProcess&& o) noexcept;
    WorkerProcess& operator=(WorkerProcess&& o) noexcept;

    /// Start `exe` (an absolute path) with `args`.  `drain_ms` is how long `finish` and the destructor wait for the
    /// worker to end after the socket is closed.
    static bool launch(const std::string& exe, const std::vector<std::string>& args, int drain_ms, WorkerProcess& out,
                       std::string& err);

    int socket() const { return fd_; }   ///< the host's end; borrowed
    pid_t pid() const { return pid_; }
    bool started() const { return pid_ > 0; }

    /// Has the worker ended?  `exit_code` (may be null) gets its exit code, or -1 when a signal ended it.
    bool ended(int* exit_code = nullptr);

    /// Close the socket and wait up to `drain_ms` for the worker to end.  True when it ended with exit code 0.
    /// A worker that is still running after the wait is reported with its PID and left alone; `finish` can be
    /// called again to go on waiting.
    bool finish(std::string& err);

private:
    int fd_ = -1;
    pid_t pid_ = -1;
    int drain_ms_ = 0;
    bool reaped_ = false;
    int status_ = 0;
    void close_socket() noexcept;
};

} // namespace strata::hetero
