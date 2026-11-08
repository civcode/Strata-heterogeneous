// Real RemoteExpertOpt teardown on two HIP devices. Report synthetic failures only
// after valid work completes; never fault a GPU. Fork/exec occurs before HIP init.
#include "strata/core/remote_experts.hpp"
#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/device.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
std::string mode;
int primary = 0, helper = 1;
bool initializing = false, armed = false;
void* mapped_owner = nullptr;
void* helper_owner = nullptr;
unsigned completed = 0;
int releases = 0;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void before_free(void* p) {
    if (!armed || (p != mapped_owner && p != helper_owner)) return;
    if (completed != 3) { std::puts("EARLY_FREE"); std::_Exit(2); }
    ++releases;
    std::puts("RELEASE");
}
struct Source : strata::core::ExpertSource {
    std::vector<uint8_t> bytes = std::vector<uint8_t>(strata::kernels::cpu::BLOB, 0);
    const uint8_t* blob(int64_t, int64_t) override { return bytes.data(); }
};
}
extern "C" {
hipError_t __real_hipDeviceSynchronize();
hipError_t __real_hipSetDevice(int);
hipError_t __real_hipHostMalloc(void**, size_t, unsigned);
hipError_t __real_hipHostGetDevicePointer(void**, void*, unsigned);
hipError_t __real_hipMalloc(void**, size_t);
hipError_t __real_hipFree(void*);
hipError_t __real_hipFreeHost(void*);
hipError_t __wrap_hipSetDevice(int d) {
    if (armed && d == helper && mode == "helper-select-failure") {
        std::puts("INJECTED"); return hipErrorUnknown;
    }
    return __real_hipSetDevice(d);
}
hipError_t __wrap_hipDeviceSynchronize() {
    int d = -1;
    if (hipGetDevice(&d) != hipSuccess) return hipErrorUnknown;
    const auto e = __real_hipDeviceSynchronize();
    if (armed && e == hipSuccess) {
        if ((d == primary && mode == "primary-sync-failure") ||
            (d == helper && mode == "helper-sync-failure")) {
            std::puts("INJECTED"); return hipErrorUnknown;
        }
        if (d == primary) completed |= 1;
        if (d == helper) completed |= 2;
    }
    return e;
}
hipError_t __wrap_hipHostMalloc(void** p, size_t n, unsigned f) {
    const auto e = __real_hipHostMalloc(p, n, f);
    if (initializing && e == hipSuccess) mapped_owner = *p;
    return e;
}
hipError_t __wrap_hipHostGetDevicePointer(void** p, void* h, unsigned f) {
    if (initializing && mode == "partial-map") return hipErrorUnknown;
    return __real_hipHostGetDevicePointer(p, h, f);
}
hipError_t __wrap_hipMalloc(void** p, size_t n) {
    if (initializing && mode == "partial-init") return hipErrorOutOfMemory;
    const auto e = __real_hipMalloc(p, n);
    if (initializing && e == hipSuccess) helper_owner = *p;
    return e;
}
hipError_t __wrap_hipFree(void* p) { before_free(p); return __real_hipFree(p); }
hipError_t __wrap_hipFreeHost(void* p) { before_free(p); return __real_hipFreeHost(p); }
}

int child_run(bool reverse) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count < 2) return 77;
    primary = reverse ? 1 : 0; helper = 1 - primary;
    strata::core::set_primary_device(primary);
    require(hipSetDevice(primary) == hipSuccess, "primary selection");
    Source source;
    strata::core::ExpertCache cache;
    strata::core::RemoteExperts remote;
    // The optimization dies before its attached helper, as in generate.cpp.
    auto opt = std::make_unique<strata::core::RemoteExpertOpt>();
    opt->attach(remote);
    std::vector<uint8_t> claimed(2, 0);
    std::string err;
    require(remote.open(helper, 1, 1, 2, {{0, 0}}, cache, source, claimed, err), err.c_str());
    initializing = true;
    const bool initialized = opt->init(err);
    initializing = false;
    const bool partial = mode == "partial-init" || mode == "partial-map";
    require(initialized != partial, "init result / injection missed");
    require(mapped_owner != nullptr && (partial || helper_owner != nullptr), "tracked optimization owners");
    constexpr int64_t H = strata::kernels::cpu::H;
    hipStream_t stream = nullptr;
    float *dst = nullptr, *rows = nullptr, *host_rows = nullptr;
    int32_t *hits = nullptr, *hit_count = nullptr;
    std::vector<float> out(H, 1), row_out(10 * H, -1);
    if (!partial) {
        std::array<int32_t, 10> ids; ids.fill(-1); ids[0] = 0;
        std::array<float, 10> weights{}; weights[0] = 1;
        std::vector<float> x(H, 1), ignored(10 * H);
        opt->begin(weights.data(), 0, 1);
        require(remote.begin(0, x.data(), ids.data(), 1, 10, nullptr, nullptr, err), err.c_str());
        require(remote.finish(ignored.data(), err), err.c_str());
        require(remote.computed() == 1 && remote.launched_layers() == 1, "optimized helper work executed");
        require(hipSetDevice(primary) == hipSuccess, "reader device");
        require(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess, "reader stream");
        require(hipMalloc(&dst, H * sizeof(float)) == hipSuccess, "output");
        require(hipMalloc(&rows, 10 * H * sizeof(float)) == hipSuccess, "rows");
        require(hipHostMalloc(&host_rows, 10 * H * sizeof(float), hipHostMallocMapped | hipHostMallocPortable) == hipSuccess, "mapped rows");
        std::fill(host_rows, host_rows + 10 * H, 2);
        float* mapped_rows = nullptr;
        require(hipHostGetDevicePointer(reinterpret_cast<void**>(&mapped_rows), host_rows, 0) == hipSuccess, "mapped row alias");
        require(hipMalloc(&hits, sizeof(int32_t)) == hipSuccess, "hits");
        require(hipMalloc(&hit_count, sizeof(int32_t)) == hipSuccess, "hit count");
        require(hipMemset(hit_count, 0, sizeof(int32_t)) == hipSuccess, "empty hits");
        require(hipMemcpy(dst, out.data(), H * sizeof(float), hipMemcpyHostToDevice) == hipSuccess, "output init");
        opt->copy_rows(rows, mapped_rows, 0, 1, hits, hit_count, stream);
        opt->combine(dst, nullptr, 0, 1, nullptr, 0, stream);
        require(hipGetLastError() == hipSuccess, "primary readers launched");
        opt->end();
    }
    // Entry differs from the primary; teardown must restore it on success.
    require(hipSetDevice(helper) == hipSuccess, "entry device");
    armed = true;
    opt.reset(); // actual production destructor, not a copied sequence
    armed = false;
    int restored = -1;
    require(hipGetDevice(&restored) == hipSuccess && restored == helper, "entry device restored");
    require(completed == 3 && releases == (partial ? 1 : 2), "all owners fenced and freed");
    require(!remote.optimized_decode(), "helper detached");
    if (!partial) {
        require(hipSetDevice(primary) == hipSuccess, "readback device");
        require(hipMemcpy(out.data(), dst, H * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess, "output readback");
        require(hipMemcpy(row_out.data(), rows, 10 * H * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess, "rows readback");
        for (float v : out) require(v == 1, "optimized zero-expert reduction result");
        for (size_t i = 0; i < row_out.size(); ++i)
            require(row_out[i] == (i < size_t(H) ? 0 : 2), "mapped mask reader result");
        require(hipFree(dst) == hipSuccess && hipFree(rows) == hipSuccess && hipFree(hits) == hipSuccess &&
                hipFree(hit_count) == hipSuccess && hipHostFree(host_rows) == hipSuccess &&
                hipStreamDestroy(stream) == hipSuccess, "fixture cleanup");
    }
    std::puts("TEARDOWN_PASS");
    return 0;
}
int main(int argc, char** argv) {
    try {
        require(argc >= 2, "scenario required");
        mode = argv[1];
        require(mode == "success" || mode == "partial-init" || mode == "partial-map" ||
                mode == "primary-sync-failure" || mode == "helper-sync-failure" ||
                mode == "helper-select-failure", "unknown scenario");
        if (argc == 4 && std::string(argv[3]) == "child") return child_run(std::string(argv[2]) == "reverse");
        int fds[2]; require(pipe(fds) == 0, "pipe");
        const pid_t pid = fork(); require(pid >= 0, "fork");
        if (pid == 0) {
            close(fds[0]);
            if (dup2(fds[1], STDOUT_FILENO) < 0) std::_Exit(3);
            close(fds[1]);
            execl(argv[0], argv[0], argv[1], argc > 2 ? argv[2] : "forward", "child", (char*) nullptr);
            std::_Exit(3);
        }
        close(fds[1]);
        std::string output; std::array<char, 1024> buf{}; ssize_t n;
        while ((n = read(fds[0], buf.data(), buf.size())) > 0) output.append(buf.data(), size_t(n));
        close(fds[0]);
        int status = 0; require(waitpid(pid, &status, 0) == pid && WIFEXITED(status), "child exit");
        std::fputs(output.c_str(), stdout);
        if (WEXITSTATUS(status) == 77) return 77;
        const bool failure = mode.find("failure") != std::string::npos;
        require(WEXITSTATUS(status) == (failure ? 1 : 0), "unexpected exit code");
        if (failure) require(output.find("INJECTED") != std::string::npos && output.find("ERR ") != std::string::npos &&
                             output.find("RELEASE") == std::string::npos && output.find("TEARDOWN_PASS") == std::string::npos,
                             "failure did not exit before owner release");
        else require(output.find("TEARDOWN_PASS") != std::string::npos, "teardown not exercised");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 2; }
}
