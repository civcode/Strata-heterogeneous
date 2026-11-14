#pragma once

// Linux host-side worker supervision.  Neither this API nor its implementation
// loads CUDA/HIP.  Executables decide which backend runtime to initialize.

#include "strata_heterogeneous/host.hpp"

#include <string>
#include <vector>
#include <sys/types.h>

namespace strata::heterogeneous {

// Owns subprocesses and their coordinator-side AF_UNIX sockets. A live
// PipelineSession may borrow descriptors through sockets(); it must not
// outlive this object. Closing sockets causes workers to observe EOF.
class WorkerProcessGroup {
public:
    WorkerProcessGroup() = default;
    ~WorkerProcessGroup() noexcept { terminate(); }
    WorkerProcessGroup(const WorkerProcessGroup&) = delete;
    WorkerProcessGroup& operator=(const WorkerProcessGroup&) = delete;
    WorkerProcessGroup(WorkerProcessGroup&& other) noexcept;
    WorkerProcessGroup& operator=(WorkerProcessGroup&& other) noexcept;

    // Each process receives only its backend's executable; a worker gets
    // a dedicated inherited socket on fd 3. Paths must be absolute.
    // Legacy CUDA config applies ONLY to CUDA children. Optional per-stage
    // paths override that mode and pass --config only to their corresponding
    // worker. An empty stage path retains unconfigured mock-worker argv.
    // Paths are validated before spawning any process; no config is read here.
    static WorkerProcessGroup launch(const Plan& plan,
                                     const std::string& cuda_executable,
                                     const std::string& hip_executable,
                                     bool shared_memory = false,
                                     const std::string& cuda_config_path = {},
                                     const std::vector<std::string>& stage_config_paths = {},
                                     bool group_hip = false);
    std::vector<int> sockets() const;
    size_t size() const { return children_.size(); }

    // Graceful closure and bounded wait; remaining children are killed/reaped.
    // False reports a non-zero exit, abnormal termination or timeout.
    // Call only after no PipelineSession operation is in flight.
    bool finish(std::string& error, int grace_ms = 1500);
    // Used for error paths/destruction. SIGKILL + waitpid avoids orphan workers.
    void terminate() noexcept;
private:
    struct Child { int fd = -1; pid_t pid = -1; };
    std::vector<Child> children_;
};

// Helpers shared with CPU mock worker executables. No GPU discovery.
std::string plan_spec(const Plan& plan);
std::string executable_directory();

} // namespace strata::heterogeneous
