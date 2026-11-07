// Actual runtime success paths. Failure ownership is tested without device faults in pinned_ring_test.
#include "strata/core/pinned_ring.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>

int main() {
    int count=0;
    if (cudaGetDeviceCount(&count)!=cudaSuccess || count==0) return 77;
    for(int dev=0; dev<count; ++dev) {
        cudaDeviceProp prop{};
        if(cudaGetDeviceProperties(&prop,dev)!=cudaSuccess) return 1;
        if(prop.integrated) continue;
        if(cudaSetDevice(dev)!=cudaSuccess) return 1;
        cudaStream_t s=nullptr;
        if(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking)!=cudaSuccess) return 1;
        void* d=nullptr;
        constexpr size_t maximum=4*1024*1024+17;
        if(cudaMalloc(&d,maximum)!=cudaSuccess) return 1;
        std::vector<unsigned char> src(maximum), dst(maximum+2,0xa5);
        for(size_t i=0;i<src.size();++i)src[i]=static_cast<unsigned char>(i*37+dev);
        {
            strata::core::PinnedRing ring;
            for(size_t n:{size_t(0),size_t(1),size_t(65536),size_t(65537),maximum}) {
                if(ring.h2d(d,src.data(),n,s)!=cudaSuccess ||
                   ring.d2h_sync(dst.data()+1,d,n,s)!=cudaSuccess ||
                   std::memcmp(src.data(),dst.data()+1,n) || dst.front()!=0xa5 || dst.back()!=0xa5) return 1;
            }
            if(!ring.reserve(maximum))return 1;
            const auto start=std::chrono::steady_clock::now();
            for(int i=0;i<100;++i) {
                if(ring.h2d(d,src.data(),65536,s)!=cudaSuccess ||
                   ring.d2h_sync(dst.data()+1,d,65536,s)!=cudaSuccess ||
                   std::memcmp(src.data(),dst.data()+1,65536))return 1;
            }
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            ring.release();
            if(ring.poisoned() || ring.quarantined_bytes())return 1;
            std::printf("GPU %d: exact ring transfers, 100 64-KiB upload/download pairs %.3f ms\n",dev,ms);
        }
        if(cudaFree(d)!=cudaSuccess || cudaStreamDestroy(s)!=cudaSuccess)return 1;
    }
    return 0;
}
