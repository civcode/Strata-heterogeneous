#pragma once
// Test-only runtime declarations. No GPU code or production failure switches.
#include <cstddef>
struct FakeEvent;
using cudaEvent_t = FakeEvent*;
using cudaStream_t = void*;
enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorInvalidValue,
                   cudaErrorInvalidDevice, cudaErrorUnknown };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
constexpr unsigned cudaHostAllocPortable = 1, cudaEventDisableTiming = 2;
cudaError_t cudaGetDevice(int*);
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaHostAlloc(void**, size_t, unsigned);
cudaError_t cudaFreeHost(void*);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
