// Two-card tests of the production PeerExperts owner and its borrowed x/out/blob
// operands. Synthetic API errors follow valid work; no GPU fault is manufactured.
#include "strata/core/peer_experts.hpp"
#include "strata/core/device.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {
constexpr int H = strata::kernels::cpu::H;
std::string mode, phase;
std::map<void*, bool> owners; // true = borrowed caller storage; false = peer-owned
bool tracking = false, teardown_complete = false;
bool outstanding = false;
int copies = 0, drains = 0, injections = 0, peer_releases = 0;
hipStream_t submitted = nullptr;
void require(bool yes, const char* why) { if (!yes) throw std::runtime_error(why); }
void inject() { ++injections; std::puts("INJECTED"); }
bool fatal() {
    return mode == "close-sync-failure" || mode == "select-failure" || mode == "issue-drain-failure" ||
           mode == "finish-drain-failure" || mode == "refill-drain-failure" ||
           mode == "refill-query-failure" || mode == "refill-wait-failure";
}
void releasing(void* p) {
    const auto it = owners.find(p);
    if (it == owners.end()) return;
    // Cache admission probes can allocate and release temporary buffers in open.
    // Keep tracking the retained owners, but do not treat probes as teardown.
    if (phase == "open" && !(mode == "partial-init" && injections > 0)) {
        if (!it->second) ++peer_releases;
        owners.erase(it);
        return;
    }
    if (outstanding || (!it->second && !teardown_complete)) {
        std::fprintf(stderr, "release phase=%s borrowed=%d outstanding=%d teardown=%d\n",
                     phase.c_str(), int(it->second), int(outstanding), int(teardown_complete));
        std::puts("EARLY_RELEASE"); std::_Exit(2);
    }
    std::puts(it->second ? "BORROWER_RELEASE" : "PEER_RELEASE");
    if (!it->second) ++peer_releases;
    owners.erase(it);
}
}
extern "C" {
hipError_t __real_hipMalloc(void**, size_t);
hipError_t __real_hipHostMalloc(void**, size_t, unsigned);
hipError_t __real_hipFree(void*);
hipError_t __real_hipFreeHost(void*);
hipError_t __real_hipMemcpyAsync(void*, const void*, size_t, hipMemcpyKind, hipStream_t);
hipError_t __real_hipStreamSynchronize(hipStream_t);
hipError_t __real_hipDeviceSynchronize();
hipError_t __real_hipSetDevice(int);
hipError_t __real_hipStreamQuery(hipStream_t);
hipError_t __real_hipEventRecord(hipEvent_t, hipStream_t);
hipError_t __real_hipEventQuery(hipEvent_t);
hipError_t __real_hipEventSynchronize(hipEvent_t);
hipError_t __wrap_hipMalloc(void** p, size_t n) {
    if (tracking && mode == "partial-init") { inject(); return hipErrorOutOfMemory; }
    const auto e = __real_hipMalloc(p, n);
    if (tracking && e == hipSuccess) owners[*p] = false;
    return e;
}
hipError_t __wrap_hipHostMalloc(void** p, size_t n, unsigned f) {
    const auto e = __real_hipHostMalloc(p, n, f);
    if (tracking && e == hipSuccess) owners[*p] = false;
    return e;
}
hipError_t __wrap_hipFree(void* p) { releasing(p); return __real_hipFree(p); }
hipError_t __wrap_hipFreeHost(void* p) { releasing(p); return __real_hipFreeHost(p); }
hipError_t __wrap_hipSetDevice(int d) {
    if (phase == "close" && mode == "select-failure" && d == 1) { inject(); return hipErrorUnknown; }
    return __real_hipSetDevice(d);
}
hipError_t __wrap_hipMemcpyAsync(void* d, const void* s, size_t n, hipMemcpyKind k, hipStream_t stream) {
    const auto e = __real_hipMemcpyAsync(d, s, n, k, stream);
    if (e == hipSuccess && (phase == "launch" || phase == "adapt")) {
        ++copies; outstanding = true; submitted = stream;
        const bool fail = phase == "launch" ?
            ((copies == 1 && (mode == "input-issue" || mode == "issue-drain-failure")) ||
             (copies == 2 && mode == "plan-issue") || (copies == 3 && mode == "results-issue")) :
            (copies == 2 && (mode == "refill-issue" || mode == "refill-drain-failure"));
        if (fail) { inject(); return hipErrorUnknown; }
    }
    return e;
}
hipError_t __wrap_hipStreamSynchronize(hipStream_t s) {
    const auto e = __real_hipStreamSynchronize(s);
    if (e == hipSuccess && s == submitted) {
        ++drains;
        if (mode == "issue-drain-failure" || mode == "finish-drain-failure" || mode == "refill-drain-failure") {
            inject(); return hipErrorUnknown;
        }
        outstanding = false;
    }
    return e;
}
hipError_t __wrap_hipDeviceSynchronize() {
    const auto e = __real_hipDeviceSynchronize();
    if (e == hipSuccess && (phase == "close" || tracking)) {
        if (phase == "close" && mode == "close-sync-failure") { inject(); return hipErrorUnknown; }
        outstanding = false; teardown_complete = true;
    }
    return e;
}
hipError_t __wrap_hipStreamQuery(hipStream_t s) {
    if (phase == "finish" && (mode == "finish-query" || mode == "finish-drain-failure")) {
        const auto e = __real_hipStreamSynchronize(s);
        if (e != hipSuccess) return e;
        inject(); return hipErrorUnknown;
    }
    const auto e = __real_hipStreamQuery(s);
    if (e == hipSuccess && s == submitted) outstanding = false;
    return e;
}
hipError_t __wrap_hipEventRecord(hipEvent_t ev, hipStream_t s) {
    const auto e = __real_hipEventRecord(ev, s);
    if (phase == "adapt" && mode == "refill-record-failure" && e == hipSuccess) { inject(); return hipErrorUnknown; }
    return e;
}
hipError_t __wrap_hipEventQuery(hipEvent_t ev) {
    if (phase == "apply" && mode == "refill-not-ready") { inject(); return hipErrorNotReady; }
    if (phase == "apply" && mode == "refill-query-failure") {
        const auto e = __real_hipEventSynchronize(ev);
        if (e != hipSuccess) return e;
        inject(); return hipErrorUnknown;
    }
    const auto e = __real_hipEventQuery(ev);
    if (e == hipSuccess) outstanding = false;
    return e;
}
hipError_t __wrap_hipEventSynchronize(hipEvent_t ev) {
    const auto e = __real_hipEventSynchronize(ev);
    if (e == hipSuccess) {
        if (phase == "apply" && mode == "refill-wait-failure") { inject(); return hipErrorUnknown; }
        outstanding = false;
    }
    return e;
}
}
namespace {
struct Source : strata::core::ExpertSource {
    uint8_t* data = nullptr;
    size_t size = 0;
    Source(size_t bytes) : size(bytes) {
        require(hipHostMalloc(&data, 4 * size, hipHostMallocPortable) == hipSuccess, "source owner");
        owners[data] = true;
        std::fill(data, data + 4 * size, uint8_t{0});
    }
    ~Source() { if (data) (void) hipHostFree(data); }
    const uint8_t* blob(int64_t, int64_t e) override {
        if (phase == "adapt" && mode == "refill-source-throws" && e == 3) {
            inject(); throw std::runtime_error("injected second source failure");
        }
        return data + size * size_t(e);
    }
};
struct Rows {
    float *x = nullptr, *out = nullptr;
    Rows() {
        require(hipHostMalloc(&x, H * sizeof(float), hipHostMallocPortable | hipHostMallocMapped) == hipSuccess, "borrowed input");
        require(hipHostMalloc(&out, 10 * H * sizeof(float), hipHostMallocPortable | hipHostMallocMapped) == hipSuccess, "borrowed output");
        owners[x] = owners[out] = true;
        std::fill(x, x + H, 1); std::fill(out, out + 10 * H, 7);
    }
    ~Rows() { if (out) (void) hipHostFree(out); if (x) (void) hipHostFree(x); }
};
std::filesystem::path metadata_path(pid_t pid) {
    return std::filesystem::temp_directory_path() / ("strata-peer-native-" + std::to_string(pid));
}
int child_run() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count < 2) return 77;
    // Exercise the configured primary=1, helper=0 pair without relying only on
    // ROCR_VISIBLE_DEVICES reorder (which still assigns primary ordinal 0).
    const bool flipped = mode == "primary-one";
    const int primary_dev = flipped ? 1 : 0;
    const int helper_dev = flipped ? 0 : 1;
    strata::core::set_primary_device(primary_dev);
    require(hipSetDevice(primary_dev) == hipSuccess, "primary selection");
    setenv("STRATA_PEER_NO_P2P", "1", 1);
    strata::kernels::cpu::NativeFmt fmt;
    std::string err;
    require(strata::kernels::cpu::native_fmt(42, 42, H, strata::kernels::cpu::FF, fmt, err), err.c_str());
    const auto dir = metadata_path(getpid());
    std::filesystem::create_directory(dir);
    { std::ofstream f(dir / "native_experts.txt"); f << "0 42 42 0 " << fmt.bytes << '\n'; require(bool(f), "layout write"); }
    require(strata::kernels::cpu::expert_layout_load(dir.string(), 1, 4, err), err.c_str());
    Source source(fmt.bytes);
    Rows rows;
    strata::core::ExpertCache primary;
    strata::core::PeerExperts peer;
    tracking = true; phase = "open"; teardown_complete = false;
    const bool opened = peer.open(helper_dev, {{0, 0}, {0, 1}}, primary, source, 1, 4, 128, 2, err);
    tracking = false;
    if (mode == "partial-init") {
        require(!opened && !peer.valid() && peer_releases > 0 && teardown_complete, "partial init cleanup");
    } else {
        require(opened && peer.valid() && peer.has(0, 0) && peer.has(0, 1), err.c_str());
        teardown_complete = false; peer_releases = 0;
        if (mode.find("refill-") == 0) {
            const float usage[] = {0, 0, 10, 9}; const int32_t res0[] = {-1, -1, -1, -1};
            phase = "adapt";
            bool threw = false, ok = false;
            try { ok = peer.adapt(usage, res0, 2, err); }
            catch (const std::runtime_error&) { threw = true; }
            const bool refused = mode == "refill-issue" || mode == "refill-record-failure" || mode == "refill-source-throws";
            require(threw == (mode == "refill-source-throws") && ok != refused, "adapt outcome");
            require(!peer.has(0, 2) && !peer.has(0, 3), "uncompleted refill not published");
            phase = "apply";
            if (refused) {
                require(drains == 1 && !outstanding, "refill refusal drained before returning");
                peer.apply_pending(true);
                require(!peer.has(0, 2) && !peer.has(0, 3), "refused refill never published");
            } else {
                if (mode == "refill-not-ready" || mode == "refill-query-failure") {
                    peer.apply_pending(false);
                    require(!peer.has(0, 2) && !peer.has(0, 3), "not-ready did not publish");
                }
                peer.apply_pending(true);
                require(peer.has(0, 2) && peer.has(0, 3) && !outstanding, "complete refill published");
            }
        } else {
            std::array<int32_t, 10> ids{}, kind{}; kind[0] = 2;
            const bool staged = mode == "success-staged" || mode == "results-issue";
            float* alias = nullptr;
            require(hipSetDevice(helper_dev) == hipSuccess, "mapped output device");
            require(hipHostGetDevicePointer(reinterpret_cast<void**>(&alias), rows.out, 0) == hipSuccess, "mapped output");
            require(hipSetDevice(primary_dev) == hipSuccess, "launch entry device");
            phase = "launch";
            const bool launched = peer.launch(0, rows.x, ids.data(), 1, 10, kind.data(), err, staged ? nullptr : alias);
            const bool refused = mode == "input-issue" || mode == "plan-issue" || mode == "results-issue";
            require(launched != refused, "launch result");
            if (refused) {
                require(drains == 1 && !outstanding && !peer.launched_direct(), "submission refusal drained");
                phase = "finish"; require(peer.finish(rows.out, err), "failed submission reset launch");
                for (int i = 0; i < 10 * H; ++i) require(rows.out[i] == 7, "refused output not published");
            } else if (mode != "close-queued" && mode != "close-sync-failure" && mode != "select-failure") {
                phase = "finish";
                const bool finished = peer.finish(rows.out, err);
                require(finished == (mode != "finish-query"), "finish result");
                require(!outstanding, "finish returned with borrowed operands pending");
                if (!finished) require(drains == 1, "query error drained");
                for (int i = 0; i < 10 * H; ++i) require(rows.out[i] == (i < H ? 0 : 7), "expert output rows");
            }
        }
        phase = "close";
        peer.close();
        require(!peer.valid() && peer_releases > 0 && teardown_complete && !outstanding, "close fenced all owners");
        int current = -1; require(hipGetDevice(&current) == hipSuccess && current == primary_dev, "close restored entry device");
        if (mode == "close-queued")
            for (int i = 0; i < 10 * H; ++i) require(rows.out[i] == (i < H ? 0 : 7), "teardown completed queued scatter");
        peer.close(); // idempotent
    }
    phase.clear();
    for (const auto& entry : owners) require(entry.second, "helper allocation leaked");
    require(mode == "success-direct" || mode == "success-staged" || mode == "primary-one" || mode == "close-queued" ||
            mode == "refill-success" || injections > 0, "fault injection missed");
    std::puts("PEER_PASS");
    return 0;
}
}
int main(int argc, char** argv) {
    try {
        require(argc >= 2, "scenario required"); mode = argv[1];
        if (argc == 3 && std::string(argv[2]) == "child") return child_run();
        int fds[2]; require(pipe(fds) == 0, "pipe");
        const pid_t pid = fork(); require(pid >= 0, "fork");
        if (pid == 0) {
            close(fds[0]); if (dup2(fds[1], STDOUT_FILENO) < 0) std::_Exit(3); close(fds[1]);
            execl(argv[0], argv[0], argv[1], "child", (char*) nullptr); std::_Exit(3);
        }
        close(fds[1]); std::string output; std::array<char, 1024> buf{}; ssize_t n;
        while ((n = read(fds[0], buf.data(), buf.size())) > 0) output.append(buf.data(), size_t(n));
        close(fds[0]); int status = 0;
        require(waitpid(pid, &status, 0) == pid && WIFEXITED(status), "child exit");
        std::error_code ignored; std::filesystem::remove_all(metadata_path(pid), ignored);
        std::fputs(output.c_str(), stdout);
        if (WEXITSTATUS(status) == 77) return 77;
        require(WEXITSTATUS(status) == (fatal() ? 1 : 0), "unexpected child status");
        if (fatal()) require(output.find("INJECTED") != std::string::npos && output.find("ERR ") != std::string::npos &&
                             output.find("RELEASE") == std::string::npos && output.find("PEER_PASS") == std::string::npos,
                             "terminal failure did not precede every owner release");
        else require(output.find("PEER_PASS") != std::string::npos && output.find("EARLY_RELEASE") == std::string::npos,
                     "production path incomplete");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 2; }
}
