#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr int cudaMemcpyHostToDevice = 1;
namespace fake_cuda {
inline int allocations=0, mallocs=0, frees=0, sets=0, syncs=0, copies=0, infos=0, device=-1;
inline size_t free_bytes=1ull<<34,total_bytes=1ull<<34;inline bool fail_info=false;
inline int fail_malloc=0, fail_set=0, fail_sync=0, fail_copy=0;
inline void reset(){mallocs=frees=sets=syncs=copies=infos=0;fail_malloc=fail_set=fail_sync=fail_copy=0;device=-1;
 free_bytes=total_bytes=1ull<<34;fail_info=false;}
}
inline const char* cudaGetErrorString(cudaError_t){return "injected CUDA failure";}
inline cudaError_t cudaSetDevice(int d){++fake_cuda::sets;if(fake_cuda::fail_set==fake_cuda::sets)return 2;fake_cuda::device=d;return cudaSuccess;}
inline cudaError_t cudaMalloc(void** p,size_t n){++fake_cuda::mallocs;if(fake_cuda::fail_malloc==fake_cuda::mallocs)return 2;
 *p=std::malloc(n);if(*p)++fake_cuda::allocations;return *p?cudaSuccess:2;}
inline cudaError_t cudaFree(void* p){if(p){--fake_cuda::allocations;++fake_cuda::frees;std::free(p);}return cudaSuccess;}
inline cudaError_t cudaDeviceSynchronize(){++fake_cuda::syncs;return fake_cuda::fail_sync==fake_cuda::syncs?2:cudaSuccess;}
inline cudaError_t cudaMemcpy(void* d,const void* s,size_t n,int){++fake_cuda::copies;
 if(fake_cuda::fail_copy==fake_cuda::copies)return 2;
 std::memcpy(d,s,n);return cudaSuccess;}

inline cudaError_t cudaMemGetInfo(size_t* f,size_t* t){++fake_cuda::infos;
 if(fake_cuda::fail_info)return 2;
 *f=fake_cuda::free_bytes;*t=fake_cuda::total_bytes;return cudaSuccess;}
