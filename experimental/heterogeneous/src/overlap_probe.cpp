// Model-free qualification only: separate CUDA/HIP builds, never mixed runtimes.
#include <cuda_runtime.h>
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <array>
#include <cstdint>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t bytes = 65536, words = 4096, flag_offset = 32768, error_offset = flag_offset + 64;
constexpr unsigned long long timeout_ns = 2000000000ull;
void ck(cudaError_t e, const char* op) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(e));
}
int number(const char* s) {
    size_t used = 0;
    const int value = std::stoi(s, &used);
    if (s[used] != '\0' || value < 0) throw std::runtime_error("invalid fd or device ordinal");
    return value;
}
float value(int seq, size_t i) { return static_cast<float>(seq * 8192 + i); }
struct Device {
    int ordinal;
    cudaStream_t stream = nullptr;
    float* data = nullptr;
    uint32_t* counter = nullptr;
    uint8_t* alias = nullptr;
    enum State { idle, prepared, armed, publishing } state = idle;
    int sequence = 0;
    float* payload() const { return reinterpret_cast<float*>(alias); }
    uint32_t* flag() const { return reinterpret_cast<uint32_t*>(alias + flag_offset); }
    uint32_t* error() const { return reinterpret_cast<uint32_t*>(alias + error_offset); }
};
struct Probe {
    uint8_t* page = nullptr;
    bool registered = false;
    std::vector<Device> devices;
    bool cleanup() noexcept {
        // Drain all streams before any registration/alias or device storage is released.
        for (auto& d : devices) {
            if (cudaSetDevice(d.ordinal) != cudaSuccess ||
                (d.stream && cudaStreamSynchronize(d.stream) != cudaSuccess)) {
                std::cerr << "UNVERIFIED native drain; no explicit resource free\n";
                return false;
            }
        }
        bool ok = true;
        for (auto& d : devices) {
            if (cudaSetDevice(d.ordinal) != cudaSuccess) { ok = false; continue; }
            if (d.data && cudaFree(d.data) != cudaSuccess) ok = false;
            if (d.counter && cudaFree(d.counter) != cudaSuccess) ok = false;
            if (d.stream && cudaStreamDestroy(d.stream) != cudaSuccess) ok = false;
        }
        if (registered) {
            if (cudaSetDevice(devices.front().ordinal) != cudaSuccess || cudaHostUnregister(page) != cudaSuccess) {
                std::cerr << "UNVERIFIED native host unregister\n";
                return false;
            }
        }
        registered = false;
        if (page && munmap(page, bytes) != 0) ok = false;
        page = nullptr;
        return ok;
    }
    void open(int fd, int argc, char** argv) {
        struct stat st{};
        if (fstat(fd, &st) != 0 || st.st_size != static_cast<off_t>(bytes))
            throw std::runtime_error("wrong shared page extent");
        auto* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) throw std::runtime_error("shared mmap failed");
        page = static_cast<uint8_t*>(p);
        int count = 0;
        ck(cudaGetDeviceCount(&count), "device count");
        for (int i = 2; i < argc; ++i) {
            const int ordinal = number(argv[i]);
            if (ordinal >= count) throw std::runtime_error("device not visible");
            for (const auto& d : devices) if (d.ordinal == ordinal) throw std::runtime_error("duplicate device");
            devices.push_back(Device{ordinal});
        }
        ck(cudaSetDevice(devices.front().ordinal), "registration owner");
        ck(cudaHostRegister(page, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable), "shared host registration");
        registered = true;
        for (auto& d : devices) {
            ck(cudaSetDevice(d.ordinal), "alias owner");
            cudaDeviceProp prop{};
            ck(cudaGetDeviceProperties(&prop, d.ordinal), "device properties");
            std::fprintf(stderr, "overlap probe ordinal=%d PCI=%04x:%02x:%02x name=%s\n",
                         d.ordinal, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, prop.name);
            ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d.alias), page, 0), "runtime-local shared alias");
            ck(cudaMalloc(reinterpret_cast<void**>(&d.data), words * sizeof(float)), "device payload");
            ck(cudaMalloc(reinterpret_cast<void**>(&d.counter), sizeof(uint32_t)), "publisher counter");
            ck(cudaMemset(d.counter, 0, sizeof(uint32_t)), "counter zero");
            ck(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking), "native stream");
            ck(cudaDeviceSynchronize(), "initialization drain");
        }
    }
    Device& find(int ordinal) {
        for (auto& d : devices) if (d.ordinal == ordinal) return d;
        throw std::runtime_error("command names unowned ordinal");
    }
    void command(const std::string& op, int ordinal, int seq) {
        auto& d = find(ordinal);
        ck(cudaSetDevice(ordinal), "command owner");
        if (seq < 1 || seq > 160) throw std::runtime_error("invalid sequence");
        auto require = [](bool ok) { if (!ok) throw std::runtime_error("probe lifecycle/sequence mismatch"); };
        if (op == "prepare") {
            require(d.state == Device::idle && seq > d.sequence);
            std::array<float, words> host{};
            for (size_t i = 0; i < words; ++i) host[i] = value(seq, i);
            ck(cudaMemcpy(d.data, host.data(), words * sizeof(float), cudaMemcpyHostToDevice), "prepare payload");
            d.sequence = seq; d.state = Device::prepared;
            std::cout << "PREPARED" << std::endl;
        } else if (op == "arm") {
            require(d.state == Device::idle && seq > d.sequence);
            d.sequence = seq; d.state = Device::armed;
            strata::kernels::wait_handoff(d.flag(), 1, d.error(), timeout_ns, d.stream);
            strata::kernels::copy_from_mapped(d.data, d.payload(), words, d.stream);
            std::cout << "ARMED" << std::endl;
        } else if (op == "pending") {
            require(d.state == Device::armed && d.sequence == seq);
            require(cudaStreamQuery(d.stream) == cudaErrorNotReady);
            std::cout << "PENDING" << std::endl;
        } else if (op == "publish") {
            require(d.state == Device::prepared && d.sequence == seq);
            d.state = Device::publishing;
            strata::kernels::handoff_publish(d.payload(), d.data, 3072, d.data+3072, 768,
                                             d.data+3840, 256, d.counter, d.flag(), d.stream);
            std::cout << "QUEUED" << std::endl;
        } else if (op == "finish") {
            require(d.state == Device::armed && d.sequence == seq);
            ck(cudaStreamSynchronize(d.stream), "consumer drain");
            require(*reinterpret_cast<volatile uint32_t*>(page + error_offset) == 0);
            std::array<float, words> host{};
            ck(cudaMemcpy(host.data(), d.data, words*sizeof(float), cudaMemcpyDeviceToHost), "consumer readback");
            for (size_t i = 0; i < words; ++i)
                if (host[i] != value(seq, i)) throw std::runtime_error("stale or incorrect GPU payload");
            d.state = Device::idle;
            std::cout << "VERIFIED" << std::endl;
        } else if (op == "drain") {
            require(d.state == Device::publishing && d.sequence == seq);
            ck(cudaStreamSynchronize(d.stream), "publisher drain");
            d.state = Device::idle;
            std::cout << "DRAINED" << std::endl;
        } else throw std::runtime_error("unknown probe command");
    }
};
} // namespace
int main(int argc, char** argv) {
    // On supervisor failure, EOF requests cleanup; a closed output pipe must
    // not terminate a worker with GPU work outstanding.
    std::signal(SIGPIPE, SIG_IGN);
    Probe probe;
    try {
#if defined(STRATA_USE_HIP)
        if (argc != 4) throw std::runtime_error("HIP probe needs shared fd and two ordinals");
#else
        if (argc != 3) throw std::runtime_error("CUDA probe needs shared fd and one ordinal");
#endif
        probe.open(number(argv[1]), argc, argv);
        std::cout << "READY" << std::endl;
        std::string op;
        while (std::cin >> op) {
            if (op == "quit") {
                for (const auto& d : probe.devices)
                    if (d.state != Device::idle) throw std::runtime_error("quit with undrained operation");
                if (!probe.cleanup()) return 2;
                std::cout << "BYE" << std::endl;
                return 0;
            }
            int device = -1, sequence = 0;
            if (!(std::cin >> device >> sequence)) throw std::runtime_error("invalid probe command");
            probe.command(op, device, sequence);
        }
        (void)probe.cleanup();
        return 2; // supervisor closed before completion
    } catch (const std::exception& e) {
        std::cerr << "overlap probe: " << e.what() << '\n';
        (void)probe.cleanup();
        return 2;
    }
}
