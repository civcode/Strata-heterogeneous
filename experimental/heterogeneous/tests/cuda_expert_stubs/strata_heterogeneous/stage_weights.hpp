#pragma once
#include <cstdint>
#include <vector>
#include <stdexcept>
namespace strata::heterogeneous {
struct StageWeightPlan{};
enum class Backend{cuda=1,hip=2};
struct Stage{Backend backend=Backend::cuda;uint32_t ordinal=0,first_layer=0,end_layer=0;};
struct Geometry{int64_t hc=2,n_embd=2560;};
struct Plan{int64_t total_layers=8;uint64_t model_fingerprint=123;Geometry geometry;
 std::vector<Stage> stages;
 void validate() const { if(stages.empty()||total_layers<=0)throw std::invalid_argument("plan invalid");
  int64_t expect=0;for(const auto& s:stages){if(s.first_layer!=expect||s.end_layer<=s.first_layer)throw std::invalid_argument("plan interval invalid");expect=s.end_layer;}
  if(expect!=total_layers)throw std::invalid_argument("plan end invalid");}
};
}
