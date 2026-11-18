// Can two processes on two GPU runtimes hand off through one shared host mapping, the way two stages of a vanilla
// layer split hand off through mapped pinned memory?  (.ai/SEAM.md, open point 2.)
//
// One source, built once for CUDA and once for HIP (the HIP compat header maps the runtime calls).  Run one process
// of each build on the same file:
//
//   truncate -s 8M /dev/shm/strata-hetero-probe
//   <cuda build>/mixed_split_shared_probe      writer /dev/shm/strata-hetero-probe 0 &
//   <hip build>/hip_mixed_split_shared_probe   reader /dev/shm/strata-hetero-probe 0
//
// (second argument: the file; third: the device ordinal in that runtime).  Then the other way round.
//
// The writer plays the front stage: a kernel fills the hand-off rows and then raises the ready word, both through the
// device address of the registered mapping.  The reader plays the next stage: its kernel is launched first, waits on
// the ready word, then reads the rows.  Each side prints what it measured; the reader checks every window's rows.
// Written without a GPU at hand: UNVERIFIED until it has been built and run.
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#define CK(call) do { const cudaError_t e_ = (call); if (e_ != cudaSuccess) { std::fprintf(stderr, "probe: %s: %s\n", #call, cudaGetErrorString(e_)); return 1; } } while (0)

namespace {

constexpr int kWindows = 200;
constexpr int kRows = 8;
constexpr int kRowFloats = 4 * 2560 + 2560 + 4;   // R (hc x n_embd), bo (n_embd), inject (hc) of the Q2_0 geometry
constexpr size_t kPage = 4096;
// page 0: the words; from page 1: the rows
constexpr size_t kReady = 0, kAck = 1, kWriterUp = 2, kReaderUp = 3;

uint32_t rd(volatile uint32_t* p) { return __atomic_load_n((uint32_t*) p, __ATOMIC_SEQ_CST); }
void wr(volatile uint32_t* p, uint32_t v) { __atomic_store_n((uint32_t*) p, v, __ATOMIC_SEQ_CST); }

bool wait_word(volatile uint32_t* p, uint32_t want, int timeout_ms) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (rd(p) < want) {
        if (std::chrono::steady_clock::now() >= until) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return true;
}

} // namespace

// The front stage's part: the rows, then the ready word (one thread: the order is the point, not the speed).
__global__ void produce(float* rows, int n, uint32_t window, volatile uint32_t* ready) {
    for (int i = 0; i < n; ++i) rows[i] = (float) (window % 1000u) + (float) (i % 1024) * 0.001f;
    *ready = window;
}

// The next stage's part: wait for the ready word (bounded), then read the rows.
__global__ void consume(volatile uint32_t* ready, uint32_t window, const float* rows, int n, float* sum, uint32_t* gave_up) {
    unsigned long long spins = 0;
    while (*ready < window) {
        if (++spins > 4000000000ull) { *gave_up = 1; return; }
    }
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += rows[i];
    *sum = s;
}

int main(int argc, char** argv) {
    if (argc < 4 || (std::strcmp(argv[1], "writer") != 0 && std::strcmp(argv[1], "reader") != 0)) {
        std::fprintf(stderr, "usage: %s writer|reader <shared file> <device>\n", argv[0]);
        return 2;
    }
    const bool writer = std::strcmp(argv[1], "writer") == 0;
    const char* who = writer ? "writer" : "reader";
    const int dev = std::atoi(argv[3]);
    const size_t row_bytes = (size_t) kRows * kRowFloats * sizeof(float);
    const size_t bytes = kPage + (row_bytes + kPage - 1) / kPage * kPage;

    const int fd = ::open(argv[2], O_RDWR);
    struct stat st {};
    if (fd < 0 || ::fstat(fd, &st) != 0 || (size_t) st.st_size < bytes) {
        std::fprintf(stderr, "probe %s: %s must exist and hold at least %zu bytes\n", who, argv[2], bytes);
        return 2;
    }
    void* base = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { std::perror("probe: mmap"); return 2; }

    int n_dev = 0;
    CK(cudaGetDeviceCount(&n_dev));
    if (dev < 0 || dev >= n_dev) { std::fprintf(stderr, "probe %s: no device %d (%d found)\n", who, dev, n_dev); return 2; }
    CK(cudaSetDevice(dev));

    // THE QUESTION: does this runtime register a shared file mapping as mapped pinned memory?
    const cudaError_t reg = cudaHostRegister(base, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
    std::printf("probe %s: register %zu bytes of the shared mapping: %s\n", who, bytes,
                reg == cudaSuccess ? "ok" : cudaGetErrorString(reg));
    if (reg != cudaSuccess) return 1;
    void* base_d = nullptr;
    CK(cudaHostGetDevicePointer(&base_d, base, 0));
    std::printf("probe %s: host %p -> device %p\n", who, base, base_d);

    volatile uint32_t* words_h = (volatile uint32_t*) base;
    volatile uint32_t* words_d = (volatile uint32_t*) base_d;
    float* rows_d = (float*) ((uint8_t*) base_d + kPage);
    const float* rows_h = (const float*) ((uint8_t*) base + kPage);
    const int n = kRows * kRowFloats;

    cudaStream_t cs;
    CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));

    if (writer) {
        wr(words_h + kReady, 0);
        wr(words_h + kAck, 0);
        wr(words_h + kWriterUp, 1);
        if (!wait_word(words_h + kReaderUp, 1, 60000)) { std::fprintf(stderr, "probe writer: no reader within 60 s\n"); return 1; }
        double total_us = 0.0;
        for (uint32_t w = 1; w <= (uint32_t) kWindows; ++w) {
            if (!wait_word(words_h + kAck, w - 1, 20000)) { std::fprintf(stderr, "probe writer: window %u: the reader did not finish the one before\n", w); return 1; }
            std::this_thread::sleep_for(std::chrono::microseconds(300));   // lets the reader launch its wait first
            const auto t0 = std::chrono::steady_clock::now();
            produce<<<1, 1, 0, cs>>>(rows_d, n, w, words_d + kReady);
            CK(cudaGetLastError());
            CK(cudaStreamSynchronize(cs));
            total_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            if (rd(words_h + kReady) != w) { std::fprintf(stderr, "probe writer: window %u: the host does not see the ready word the GPU wrote\n", w); return 1; }
        }
        if (!wait_word(words_h + kAck, kWindows, 20000)) { std::fprintf(stderr, "probe writer: the reader did not finish\n"); return 1; }
        std::printf("probe writer: ok, %d windows, %.1f us per publish (one-thread kernel, not a speed figure)\n", kWindows, total_us / kWindows);
    } else {
        float* sum_h = nullptr;
        uint32_t* gave_h = nullptr;
        CK(cudaHostAlloc((void**) &sum_h, 64, cudaHostAllocMapped));
        CK(cudaHostAlloc((void**) &gave_h, 64, cudaHostAllocMapped));
        float* sum_d = nullptr;
        uint32_t* gave_d = nullptr;
        CK(cudaHostGetDevicePointer((void**) &sum_d, sum_h, 0));
        CK(cudaHostGetDevicePointer((void**) &gave_d, gave_h, 0));
        if (!wait_word(words_h + kWriterUp, 1, 60000)) { std::fprintf(stderr, "probe reader: no writer within 60 s\n"); return 1; }
        wr(words_h + kReaderUp, 1);
        int bad = 0;
        double total_us = 0.0;
        for (uint32_t w = 1; w <= (uint32_t) kWindows; ++w) {
            *sum_h = -1.0f;
            *gave_h = 0;
            const auto t0 = std::chrono::steady_clock::now();
            consume<<<1, 1, 0, cs>>>(words_d + kReady, w, rows_d, n, sum_d, gave_d);   // launched before the rows exist
            CK(cudaGetLastError());
            CK(cudaStreamSynchronize(cs));
            total_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            if (*gave_h != 0) { std::fprintf(stderr, "probe reader: window %u: the GPU never saw the ready word\n", w); return 1; }
            double want = 0.0;   // the same sum on the host, from the host's view of the rows
            float s = 0.0f;
            for (int i = 0; i < n; ++i) s += rows_h[i];
            for (int i = 0; i < n; ++i) want += (double) ((float) (w % 1000u) + (float) (i % 1024) * 0.001f);
            const bool host_ok = std::abs((double) s - want) <= 1e-3 * want + 1.0;
            const bool gpu_ok = std::abs((double) *sum_h - (double) s) <= 1e-3 * std::abs((double) s) + 1.0;
            if (!host_ok || !gpu_ok) {
                if (bad < 5) std::fprintf(stderr, "probe reader: window %u: rows %s on the host, GPU sum %g vs host sum %g\n", w, host_ok ? "right" : "WRONG", (double) *sum_h, (double) s);
                ++bad;
            }
            wr(words_h + kAck, w);
        }
        if (bad) { std::fprintf(stderr, "probe reader: %d of %d windows read stale or wrong rows\n", bad, kWindows); return 1; }
        std::printf("probe reader: ok, %d windows, %.1f us per window from launch to rows read (includes the writer's pause)\n", kWindows, total_us / kWindows);
    }
    CK(cudaStreamDestroy(cs));
    CK(cudaHostUnregister(base));
    (void) ::munmap(base, bytes);
    (void) ::close(fd);
    return 0;
}
