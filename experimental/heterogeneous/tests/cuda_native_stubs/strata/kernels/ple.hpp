#pragma once
#include "strata/kernels/ngram.hpp"
#include <cstddef>
#include <cstdint>
namespace strata::kernels {
struct PleWeights {
    const uint8_t* key_codes=nullptr;
    const float* key_scales=nullptr;
    const uint16_t* key_bf16=nullptr;
    const uint16_t* value_bf16=nullptr;
    const float *norm_key=nullptr,*norm_query=nullptr,*norm_conv=nullptr;
    const uint16_t* conv1d_f16=nullptr;
    const void* key_native_data=nullptr;
    int key_native_type=-1;
    void* key_native_q8_1=nullptr;
};
}
namespace strata::core {inline uint64_t ple_run_scratch_bytes(){return 4096;} }
