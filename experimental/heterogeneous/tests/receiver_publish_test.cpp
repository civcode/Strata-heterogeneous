#include "strata_heterogeneous/receiver_publish.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
using strata::heterogeneous::publish_receiver_input;
int main() {
    std::array<uint8_t, 8> dst{}, input{1,2,3,4,5,6,7,8};
    volatile uint32_t ready = 0;
    const auto check=[](bool ok){ if (!ok) throw std::runtime_error("receiver publication contract"); };
    check(!publish_receiver_input(nullptr,dst.size(),&ready,input.data(),input.size()));
    check(!publish_receiver_input(dst.data(),dst.size(),nullptr,input.data(),input.size()));
    check(!publish_receiver_input(dst.data(),dst.size(),&ready,nullptr,input.size()));
    check(!publish_receiver_input(dst.data(),dst.size(),&ready,input.data(),0));
    check(!publish_receiver_input(dst.data(),dst.size()-1,&ready,input.data(),input.size()));
    check(ready==0 && dst==std::array<uint8_t,8>{});
    check(publish_receiver_input(dst.data(),dst.size(),&ready,input.data(),input.size()));
    check(ready==1 && dst==input);
    const auto before=dst;
    input.fill(9);
    check(!publish_receiver_input(dst.data(),dst.size(),&ready,input.data(),input.size()));
    check(dst==before && ready==1);
    ready=0; // caller has drained and admitted another window
    check(publish_receiver_input(dst.data(),dst.size(),&ready,input.data(),3));
    check(ready==1 && dst[0]==9 && dst[1]==9 && dst[2]==9 && dst[3]==4);
    std::cout << "PASS 12 receiver publication assertions (CPU only)\n";
}
