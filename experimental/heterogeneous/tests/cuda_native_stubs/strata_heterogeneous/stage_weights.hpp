#pragma once
#include "strata_heterogeneous/host.hpp"
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
namespace strata::heterogeneous {
struct StageWeightPlan {
    std::set<std::string> skip;
    uint64_t full_pool_bytes=0;
    uint64_t selected_pool_bytes=0;
    size_t total_tensors=0,selected_tensors=0,stage_layer_tensors=0;
};
bool plan_stage_weights(const Plan&,size_t,const std::string&,StageWeightPlan&,std::string&);
}
