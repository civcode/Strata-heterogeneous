// Test-only fake engine; returns a real nonzero error when hook refuses a stream.
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
int main() {
    const int calls = std::getenv("PHASE11B_FAKE_CALLS") ? std::atoi(std::getenv("PHASE11B_FAKE_CALLS")) : 1;
    for (int i = 0; i < calls; ++i) {
        hipStream_t s = nullptr;
        hipError_t rc = hipStreamCreateWithFlags(&s, 1u);
        if (rc != hipSuccess) {
            std::fprintf(stderr, "strata generate: layer split, CUDA1: its streams failed\n");
            return 1;
        }
    }
    std::printf("READY 1024\n");
    return 0;
}
