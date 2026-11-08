// Stub HIP symbols for host-only interposer tests, no GPU required.
#include <hip/hip_runtime_api.h>
#include <cstdlib>
#include <cstdint>
extern "C" hipError_t hipGetDevice(int* d) {
    *d = std::getenv("PHASE11B_FAKE_DEVICE") ? std::atoi(std::getenv("PHASE11B_FAKE_DEVICE")) : 1;
    return hipSuccess;
}
extern "C" hipError_t hipStreamCreateWithFlags(hipStream_t* s, unsigned int) {
    *s = reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x1234));
    return hipSuccess;
}
