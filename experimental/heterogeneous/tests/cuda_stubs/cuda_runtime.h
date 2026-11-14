#pragma once
// CPU test double only. Never on a production include path.
#include <cstdlib>
#include <cstring>
using cudaError_t = int;
using cudaStream_t = void*;
using cudaEvent_t = void*;
constexpr int cudaSuccess = 0, cudaErrorNotReady = 1;
constexpr int cudaEventDisableTiming = 2, cudaMemcpyHostToDevice = 3, cudaMemcpyDeviceToHost = 4;
namespace fake_cuda {
inline int allocations = 0, copies = 0, records = 0;
inline int failure = 0, query = 0, fail_alloc = 0, alloc_calls = 0;
inline bool fail_download = false;
inline int device = 0;
inline int device_syncs = 0;
}
inline const char* cudaGetErrorString(int) { return "injected CUDA error"; }
inline int cudaSetDevice(int d) { fake_cuda::device = d; return 0; }
inline int cudaDeviceSynchronize() { ++fake_cuda::device_syncs; return 0; }
inline int cudaStreamSynchronize(cudaStream_t) { return 0; }
inline int cudaMemGetInfo(size_t* f,size_t* t) { *f = *t = 1ull<<30; return 0; }
inline int cudaMalloc(void** p, size_t n) {
    if (++fake_cuda::alloc_calls == fake_cuda::fail_alloc) return 2;
    *p = std::malloc(n); if (*p) ++fake_cuda::allocations; return *p ? 0 : 2;
}
inline int cudaMallocHost(void** p, size_t n) { return cudaMalloc(p, n); }
constexpr unsigned cudaHostAllocMapped=8, cudaHostAllocPortable=16;
inline int cudaHostAlloc(void** p,size_t n,unsigned) { return cudaMallocHost(p,n); }
inline int cudaHostGetDevicePointer(void** d,void* h,unsigned) { *d=h; return 0; }
inline int cudaFree(void* p) { if (p) { --fake_cuda::allocations; std::free(p); } return 0; }
inline int cudaFreeHost(void* p) { return cudaFree(p); }
inline int cudaEventCreateWithFlags(void** p, unsigned) { *p = reinterpret_cast<void*>(1); return 0; }
inline int cudaEventDestroy(void*) { return 0; }
inline int cudaEventRecord(void*, void*) { ++fake_cuda::records; return 0; }
inline int cudaEventQuery(void*) { return fake_cuda::query; }
inline int cudaStreamQuery(void*) { return 0; }
inline int cudaMemcpyAsync(void* dst, const void* src, size_t n, int kind, void*) {
    ++fake_cuda::copies;
    if (fake_cuda::failure || (fake_cuda::fail_download && kind == cudaMemcpyDeviceToHost)) return 2;
    std::memcpy(dst, src, n); return 0;
}
