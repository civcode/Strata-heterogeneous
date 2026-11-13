// CPU-only startup/identity preflight. This is not a HIP inference executor.
// In supervised mode it always fails closed; no HIP runtime or model is loaded.
#include "strata_heterogeneous/hip_worker_bootstrap.hpp"
#include "strata_heterogeneous/hip_worker_config.hpp"
#include "strata_heterogeneous/host.hpp"
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>

namespace hetero = strata::heterogeneous;
namespace {
bool socket_stream(int fd) {
    int type = 0;
    socklen_t len = sizeof(type);
    return ::getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &len) == 0 &&
           type == SOCK_STREAM;
}
void report_failure(int fd, const hetero::Plan& plan, size_t index,
                    const std::string& reason) noexcept {
    if (!socket_stream(fd) || index >= plan.stages.size()) return;
    try {
        hetero::Frame f;
        f.kind = hetero::Kind::failure;
        f.stage_index = static_cast<uint32_t>(index);
        f.request_id = 1; f.window_id = 1; f.rows = 1;
        f.plan_fingerprint = plan.fingerprint();
        f.geometry = plan.geometry;
        f.error = reason.substr(0, 1024);
        hetero::send_frame(fd, f, 1000);
    } catch (...) {} // caller may have disconnected
}
} // namespace
int main(int argc, char** argv) {
    try {
        std::string error;
        if (argc == 3 && std::string(argv[1]) == "--check-config") {
            hetero::HipWorkerConfig config;
            if (!hetero::read_hip_worker_config(argv[2], config, error) ||
                !hetero::validate_cuda_worker_files(config.fields, error))
                throw std::runtime_error(error);
            std::cout << "PASS: HIP v2 config/pack/GGUF preflight (no HIP runtime), stage "
                      << config.fields.stage_index << '\n';
            return 0;
        }
        hetero::HipWorkerInvocation invocation;
        if (!hetero::parse_hip_worker_invocation(
                argc, const_cast<const char* const*>(argv), invocation, error))
            throw std::runtime_error(error);
        if (!socket_stream(3))
            throw std::runtime_error("HIP preflight requires supervisor SOCK_STREAM on fd 3");
        try {
            hetero::HipWorkerConfig config;
            if (!hetero::read_hip_worker_config(invocation.config_path, config, error) ||
                !hetero::validate_hip_worker_config(config, invocation.plan,
                                                     invocation.stage_index, error))
                throw std::runtime_error(error);
            // Refuse even a valid config; this binary MUST NOT advertise ready.
            throw std::runtime_error("HIP preflight-only worker: HIP inference adapter not implemented");
        } catch (const std::exception& ex) {
            report_failure(3, invocation.plan, invocation.stage_index, ex.what());
            std::cerr << "HIP preflight: " << ex.what() << '\n';
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "HIP preflight: " << ex.what() << '\n';
        return 2;
    }
}
