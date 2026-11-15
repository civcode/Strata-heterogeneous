// Explicit model-free qualification. Separate CUDA and HIP native builds.
// No model, new GPU math, cross-runtime handle, or CPU activation relay.
#include "strata_heterogeneous/direct_handoff_registration.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/platform/cpu_relax.hpp"
#include <atomic>
#include <charconv>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strata::heterogeneous;
using strata::kernels::copy_from_mapped;
using strata::kernels::handoff_publish;
using strata::kernels::wait_handoff;

constexpr uint32_t max_sequence = 65536;

int arg_int(const char* s, int lo, int hi) {
    int v = 0;
    const char* end = s + std::strlen(s);
    auto parsed = std::from_chars(s, end, v);
    if (parsed.ec != std::errc{} || parsed.ptr != end || v < lo || v > hi)
        throw std::invalid_argument("invalid direct probe numeric argument");
    return v;
}
void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}
uint32_t read_word(const uint32_t* p) {
    const auto value = *static_cast<const volatile uint32_t*>(p);
    strata_load_fence();
    std::atomic_thread_fence(std::memory_order_acquire);
    return value;
}
void reset_words(DirectHandoffView v) {
    *static_cast<volatile uint32_t*>(v.ready()) = 0;
    *static_cast<volatile uint32_t*>(v.error()) = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    strata_store_fence();
}

struct NativeGraph {
    int device = -1;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    float* data = nullptr;
    uint32_t* counter = nullptr;

    NativeGraph() = default;
    NativeGraph(const NativeGraph&) = delete;
    NativeGraph& operator=(const NativeGraph&) = delete;

    void select() const { direct_check(cudaSetDevice(device), "select graph owner"); }
    void init(int ordinal, const DirectHandoffLayout& layout, bool publishes) {
        device = ordinal;
        select();
        cudaDeviceProp prop{};
        direct_check(cudaGetDeviceProperties(&prop, device), "graph owner properties");
        std::fprintf(stderr, "direct graph ordinal=%d PCI=%04x:%02x:%02x name=%s\n",
                     device, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, prop.name);
        direct_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create graph stream");
        direct_check(cudaMalloc(reinterpret_cast<void**>(&data), layout.payload_bytes), "graph device payload");
        direct_check(cudaMemset(data, 0, layout.payload_bytes), "initial device payload zero");
        if (publishes) {
            direct_check(cudaMalloc(reinterpret_cast<void**>(&counter), sizeof(uint32_t)), "publisher block counter");
            direct_check(cudaMemset(counter, 0, sizeof(uint32_t)), "publisher block counter zero");
        }
        direct_check(cudaDeviceSynchronize(), "initial allocation drain");
    }
    template<class F> void capture(F&& kernels) {
        select();
        direct_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin native graph capture");
        try { kernels(); }
        catch (...) {
            (void)cudaStreamEndCapture(stream, &graph);
            throw;
        }
        direct_check(cudaStreamEndCapture(stream, &graph), "end native graph capture");
        require(graph != nullptr, "native graph capture produced no graph");
        direct_check(cudaGraphInstantiate(&exec, graph, 0ull), "instantiate native graph");
    }
    void publish(DirectHandoffView dst) {
        const auto& l = dst.layout;
        handoff_publish(dst.payload(), data, static_cast<int64_t>(l.r_words),
                        data + l.r_words, static_cast<int64_t>(l.bo_words),
                        data + l.r_words + l.bo_words, static_cast<int64_t>(l.inj_words),
                        counter, dst.ready(), stream, nullptr);
    }
    void consume(DirectHandoffView src, unsigned long long timeout_ns) {
        const auto& l = src.layout;
        // Exactly the vanilla native ready-word wait and segmented R/bo/inj reads.
        wait_handoff(src.ready(), 1u, src.error(), timeout_ns, stream);
        copy_from_mapped(data, src.payload(), static_cast<int64_t>(l.r_words), stream);
        copy_from_mapped(data + l.r_words, src.payload() + l.r_words,
                         static_cast<int64_t>(l.bo_words), stream);
        copy_from_mapped(data + l.r_words + l.bo_words, src.payload() + l.r_words + l.bo_words,
                         static_cast<int64_t>(l.inj_words), stream);
    }
    void launch() {
        select();
        direct_check(cudaGraphLaunch(exec, stream), "launch native captured graph");
    }
    void pending() {
        select();
        const auto s = cudaStreamQuery(stream);
        if (s != cudaErrorNotReady)
            throw std::runtime_error(std::string("expected pending native graph before publication; got ") + cudaGetErrorString(s));
    }
    void drain() {
        select();
        direct_check(cudaStreamSynchronize(stream), "native graph drain");
    }
    void fill(const DirectHandoffLayout& layout, uint32_t seq) {
        select();
        std::vector<float> values(layout.words());
        for (size_t i = 0; i < values.size(); ++i) values[i] = direct_probe_value(seq, i);
        direct_check(cudaMemcpy(data, values.data(), layout.payload_bytes, cudaMemcpyHostToDevice), "seed publisher data");
        direct_check(cudaDeviceSynchronize(), "seed publisher drain");
    }
    void verify(const DirectHandoffLayout& layout, uint32_t seq) {
        select();
        // Readback is qualification evidence ONLY; not part of the activation handoff.
        std::vector<float> values(layout.words());
        direct_check(cudaMemcpy(values.data(), data, layout.payload_bytes, cudaMemcpyDeviceToHost), "qualification readback");
        for (size_t i = 0; i < values.size(); ++i)
            if (values[i] != direct_probe_value(seq, i))
                throw std::runtime_error("stale or incorrect native graph payload at word " + std::to_string(i));
    }
    bool safe_to_release() const noexcept {
        return device < 0 || (cudaSetDevice(device) == cudaSuccess &&
                              (!stream || cudaStreamSynchronize(stream) == cudaSuccess));
    }
    bool release_after_drain() noexcept {
        if (device < 0) return true;
        if (cudaSetDevice(device) != cudaSuccess) return false;
        // Captured graph pointers die before buffers are released.
        if (exec && cudaGraphExecDestroy(exec) != cudaSuccess) return false;
        exec = nullptr;
        if (graph && cudaGraphDestroy(graph) != cudaSuccess) return false;
        graph = nullptr;
        if (stream && cudaStreamDestroy(stream) != cudaSuccess) return false;
        stream = nullptr;
        if (counter && cudaFree(counter) != cudaSuccess) return false;
        counter = nullptr;
        if (data && cudaFree(data) != cudaSuccess) return false;
        data = nullptr;
        device = -1;
        return true;
    }
};

class Probe {
public:
    DirectHandoffLayout layout{};
    DirectHandoffRegistration cross;
    NativeGraph first, second;
    unsigned long long timeout_ns = 0;
    uint32_t sequence = 0;
    enum class State { idle, prepared, launched, armed, poisoned } state = State::idle;
#if defined(STRATA_USE_HIP)
    void* local_page = nullptr;
    DirectHandoffView local_host{};
#endif

    void open(int fd, int d0, int d1, int rows, int timeout_ms) {
        layout = DirectHandoffLayout::make(4, 2560, static_cast<size_t>(rows));
        timeout_ns = static_cast<unsigned long long>(timeout_ms) * 1000000ull;
        int count = 0;
        direct_check(cudaGetDeviceCount(&count), "device inventory");
        require(d0 >= 0 && d0 < count, "first device not visible");
#if defined(STRATA_USE_HIP)
        require(d1 >= 0 && d1 < count && d1 != d0, "invalid HIP pair");
#else
        require(d1 == -1, "CUDA probe owns exactly one GPU");
#endif
        cross.open(fd, d0, layout);
        const auto cross_alias = cross.local_alias(d0);
        first.init(d0, layout, true);
#if defined(STRATA_USE_HIP)
        second.init(d1, layout, false);
        first.select();
        // Same runtime-local coherent mapped allocation as vanilla HIP split overlap.
        direct_check(cudaHostAlloc(&local_page, layout.extent,
                                   cudaHostAllocMapped | cudaHostAllocPortable | hipHostMallocCoherent),
                     "allocate native HIP-to-HIP handoff");
        local_host = {static_cast<uint8_t*>(local_page), layout};
        std::memset(local_page, 0, layout.extent);
        void *a = nullptr, *b = nullptr;
        direct_check(cudaHostGetDevicePointer(&a, local_page, 0), "HIP publisher alias");
        second.select();
        direct_check(cudaHostGetDevicePointer(&b, local_page, 0), "HIP consumer alias");
        require(a && b, "null HIP-local aliases");
        const DirectHandoffView local_out{static_cast<uint8_t*>(a), layout};
        const DirectHandoffView local_in{static_cast<uint8_t*>(b), layout};
        // Capture both downstream graphs before any live waiting kernel starts.
        first.capture([&] { first.consume(cross_alias, timeout_ns); first.publish(local_out); });
        second.capture([&] { second.consume(local_in, timeout_ns); });
#else
        first.capture([&] { first.publish(cross_alias); });
#endif
    }
    bool cleanup() noexcept {
        // All streams first, then graphs, then their captured host registrations.
        // Never explicitly free memory if device completion is uncertain.
        const bool a = first.safe_to_release();
        const bool b = second.safe_to_release();
        if (!a || !b) {
            std::cerr << "UNVERIFIED native drain; retaining mappings until process exit\n";
            return false;
        }
#if defined(STRATA_USE_HIP)
        const int local_owner = first.device;
#endif
        if (!first.release_after_drain() || !second.release_after_drain()) return false;
#if defined(STRATA_USE_HIP)
        if (local_page) {
            if (cudaSetDevice(local_owner) != cudaSuccess || cudaFreeHost(local_page) != cudaSuccess) return false;
            local_page = nullptr;
        }
#endif
        return cross.release_after_drain();
    }
    void command(const std::string& op, uint32_t seq) {
        require(seq > 0 && seq <= max_sequence, "sequence out of range");
        const auto host = cross.host();
        auto next = [&](State s) {
            require(state == State::idle && seq > sequence, "stale or overlapping graph round");
            sequence = seq; state = s;
        };
        auto same = [&](State s) {
            require(state == s && seq == sequence, "graph round/state mismatch");
        };
#if defined(STRATA_USE_HIP)
        if (op == "arm") {
            next(State::armed);
            require(read_word(host.ready()) == 0 && read_word(host.error()) == 0, "cross-edge was not reset");
            reset_words(local_host);
            // Prelaunch terminal before intermediate; no CUDA publication yet.
            second.launch(); first.launch();
            std::cout << "ARMED" << std::endl;
        } else if (op == "pending") {
            same(State::armed);
            first.pending(); second.pending();
            require(!read_word(host.ready()) && !read_word(host.error()) &&
                    !read_word(local_host.ready()) && !read_word(local_host.error()), "unexpected prepublication flag/error");
            std::cout << "PENDING" << std::endl;
        } else if (op == "finish") {
            same(State::armed);
            first.drain(); second.drain();
            require(read_word(host.ready()) == 1 && read_word(host.error()) == 0 &&
                    read_word(local_host.ready()) == 1 && read_word(local_host.error()) == 0, "handoff flag/error gate failed");
            first.verify(layout, seq); second.verify(layout, seq);
            state = State::idle;
            std::cout << "VERIFIED" << std::endl;
        } else if (op == "reject") {
            same(State::armed);
            // Native timeout still allows later graph nodes to drain; the data
            // is NOT used. Fail-stop this entire session, including both HIP GPUs.
            first.drain(); second.drain();
            require(read_word(host.ready()) == 0 && read_word(host.error()) == 1, "missing publication did not fail closed");
            state = State::poisoned;
            std::cout << "REJECTED" << std::endl;
        } else throw std::runtime_error("unknown HIP graph probe command");
#else
        if (op == "prepare") {
            next(State::prepared);
            reset_words(host); // supervisor ordered this only after previous BOTH drains.
            first.fill(layout, seq);
            std::cout << "PREPARED" << std::endl;
        } else if (op == "publish") {
            same(State::prepared);
            state = State::launched;
            first.launch();
            std::cout << "QUEUED" << std::endl;
        } else if (op == "drain") {
            same(State::launched);
            first.drain();
            require(read_word(host.ready()) == 1 && read_word(host.error()) == 0, "CUDA publication gate failed");
            state = State::idle;
            std::cout << "DRAINED" << std::endl;
        } else if (op == "reject") {
            same(State::prepared);
            require(read_word(host.ready()) == 0 && read_word(host.error()) == 1, "missing HIP rejection acknowledgment");
            state = State::poisoned;
            std::cout << "REJECTED" << std::endl;
        } else throw std::runtime_error("unknown CUDA graph probe command");
#endif
    }
};
} // namespace

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);
    Probe p;
    try {
        // fd, rows, timeout_ms, runtime-local ordinal(s)
#if defined(STRATA_USE_HIP)
        require(argc == 6, "HIP direct graph probe requires fd rows timeout_ms hip0 hip1");
        const int d1 = arg_int(argv[5], 0, std::numeric_limits<int>::max());
#else
        require(argc == 5, "CUDA direct graph probe requires fd rows timeout_ms cuda0");
        const int d1 = -1;
#endif
        p.open(arg_int(argv[1], 0, std::numeric_limits<int>::max()),
               arg_int(argv[4], 0, std::numeric_limits<int>::max()), d1,
               arg_int(argv[2], 1, 8), arg_int(argv[3], 100, 15000));
        std::cout << "READY " << p.layout.payload_bytes << ' ' << p.layout.extent << std::endl;
        std::string op;
        while (std::cin >> op) {
            if (op == "quit") {
                require(p.state == Probe::State::idle || p.state == Probe::State::poisoned,
                        "quit with outstanding graph work");
                if (!p.cleanup()) return 2;
                std::cout << "BYE" << std::endl;
                return 0;
            }
            std::string raw;
            if (!(std::cin >> raw)) throw std::runtime_error("missing round sequence");
            const uint32_t seq = static_cast<uint32_t>(arg_int(raw.c_str(), 1, max_sequence));
            p.command(op, seq);
        }
        (void)p.cleanup();
        return 2; // EOF is an unsuccessful qualification, even with safe drain.
    } catch (const std::exception& e) {
        std::cerr << "direct graph probe: " << e.what() << '\n';
        (void)p.cleanup();
        return 2;
    }
}
