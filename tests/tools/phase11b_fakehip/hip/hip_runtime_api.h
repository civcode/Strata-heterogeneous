#pragma once
// Header for CPU-only interposer ABI tests. NOT used by HIP production builds.
#ifdef __cplusplus
extern "C" {
#endif
typedef void* hipStream_t;
typedef enum hipError_t {
    hipSuccess = 0, hipErrorInvalidValue = 1, hipErrorOutOfMemory = 2
} hipError_t;
hipError_t hipGetDevice(int*);
hipError_t hipStreamCreateWithFlags(hipStream_t*, unsigned int);
#ifdef __cplusplus
}
#endif
