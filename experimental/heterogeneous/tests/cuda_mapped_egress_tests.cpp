#include "strata_heterogeneous/cuda_mapped_egress.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
using strata::heterogeneous::CudaMappedEgress;
int main() {
    int assertions=0;
    const auto check=[&](bool ok){ ++assertions; if (!ok) throw std::runtime_error("mapped egress contract"); };
    std::array<float, 8> out{}, expected{1,2,3,4,5,6,7,8};
    CudaMappedEgress mapped;
    check(!mapped.allocate(0));
    check(!mapped.copy_completed(out.data(),sizeof(out)));
    check(mapped.allocate(sizeof(out)));
    check(fake_cuda::allocations==2 && mapped.device_payload()!=nullptr && mapped.device_flag()!=nullptr);
    check(!mapped.allocate(sizeof(out)));
    check(!mapped.copy_completed(out.data(),sizeof(out)));
    std::memcpy(mapped.device_payload(),expected.data(),sizeof(expected));
    *mapped.device_flag()=1;
    check(mapped.copy_completed(out.data(),sizeof(out)) && out==expected);
    check(!mapped.copy_completed(out.data(),sizeof(out)+1));
    check(!mapped.copy_completed(nullptr,sizeof(out)));
    check(mapped.reset() && *mapped.host_flag()==0);
    check(!mapped.copy_completed(out.data(),sizeof(out)));
    *mapped.device_flag()=UINT32_MAX;
    check(!mapped.copy_completed(out.data(),sizeof(out)));
    check(mapped.release() && fake_cuda::allocations==0);
    check(mapped.release() && !mapped.device_payload());
    for (int i : {1,2}) {
        fake_cuda::fail_alloc=fake_cuda::alloc_calls+i;
        check(!mapped.allocate(sizeof(out)));
        fake_cuda::fail_alloc=0;
        check(mapped.release() && fake_cuda::allocations==0);
    }
    std::cout << "PASS " << assertions << " mapped egress CPU assertions\n";
}
