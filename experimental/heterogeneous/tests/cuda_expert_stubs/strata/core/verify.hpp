#pragma once
#include <cstdint>
#include <string>
namespace strata::core {
struct ModelGeometry {int64_t n_layers=8,hc=2,n_embd=2560,n_expert=512,n_vocab=100;};
struct WeightTable{};
struct SessionState{};
struct VerifyHits{const int32_t* d_res=nullptr;const int32_t* h_res=nullptr;
 const uint8_t* cache_base=nullptr;const uint64_t* slot_off=nullptr;int64_t n_slots=0,blob=0;};
using PoolMultiFn=void(*)(void*,const float*,const int32_t*,int64_t,int64_t,float*,int64_t);
}
