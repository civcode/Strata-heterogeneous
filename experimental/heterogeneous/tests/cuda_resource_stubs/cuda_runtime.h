#pragma once
#include <cstddef>
#include <cstdlib>
using cudaError_t = int;
using cudaStream_t = void*;
using cudaEvent_t = void*;
constexpr cudaError_t cudaSuccess = 0;
namespace fake_cuda {
inline int allocations=0, malloc_calls=0, free_calls=0, set_calls=0, sync_calls=0, info_calls=0;
inline int fail_malloc_at=0, fail_set_at=0, fail_sync_at=0;
inline bool fail_meminfo=false;
inline size_t free_bytes=1ull<<34, total_bytes=1ull<<34;
inline int current_device=-1;
inline void reset() {malloc_calls=free_calls=set_calls=sync_calls=info_calls=0;
    fail_malloc_at=fail_set_at=fail_sync_at=0;fail_meminfo=false;
    free_bytes=total_bytes=1ull<<34;current_device=-1;}
}
inline const char* cudaGetErrorString(cudaError_t){return "injected CUDA failure";}
inline cudaError_t cudaSetDevice(int n) { ++fake_cuda::set_calls;
    if (fake_cuda::set_calls==fake_cuda::fail_set_at) return 2;
    fake_cuda::current_device=n; return cudaSuccess; }
inline cudaError_t cudaMemGetInfo(size_t* free,size_t* total) {++fake_cuda::info_calls;
    if (fake_cuda::fail_meminfo) return 2;
    *free = fake_cuda::free_bytes;
    *total = fake_cuda::total_bytes;
    return cudaSuccess;
}
inline cudaError_t cudaMalloc(void** p,size_t n) { ++fake_cuda::malloc_calls;
    if(fake_cuda::malloc_calls==fake_cuda::fail_malloc_at) return 2;
    *p=std::malloc(n);if(*p)++fake_cuda::allocations;return *p?cudaSuccess:2; }
inline cudaError_t cudaFree(void* p) {if(p){--fake_cuda::allocations;++fake_cuda::free_calls;std::free(p);}return cudaSuccess;}
inline cudaError_t cudaDeviceSynchronize() {++fake_cuda::sync_calls;
    return fake_cuda::sync_calls==fake_cuda::fail_sync_at?2:cudaSuccess;}
