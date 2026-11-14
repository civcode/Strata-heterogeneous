#pragma once
#include <cstddef>
using cudaStream_t = void*;
using cudaEvent_t = void*;
enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorInvalidValue, cudaErrorUnknown };
enum cudaMemcpyKind { cudaMemcpyDefault, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
constexpr unsigned cudaHostAllocPortable = 1, cudaStreamNonBlocking = 1, cudaEventDisableTiming = 2;
cudaError_t cudaGetDevice(int*);
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaHostAlloc(void**, size_t, unsigned);
cudaError_t cudaFreeHost(void*);
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
