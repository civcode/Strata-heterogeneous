#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
namespace strata::heterogeneous {
enum class Backend : uint8_t { cuda=1, hip=2 };
struct Stage { Backend backend=Backend::cuda; uint32_t ordinal=0,first_layer=0,end_layer=0; };
struct Geometry { uint32_t hc=0,n_embd=0; };
struct Plan {uint32_t total_layers=0;uint64_t model_fingerprint=0;Geometry geometry{};std::vector<Stage> stages;
void validate() const {if(!total_layers||!model_fingerprint||!geometry.hc||!geometry.n_embd||stages.empty())throw std::invalid_argument("invalid plan");
uint32_t next=0;for(auto s:stages){if(s.first_layer!=next||s.end_layer<=next||s.end_layer>total_layers)throw std::invalid_argument("partition");next=s.end_layer;}if(next!=total_layers)throw std::invalid_argument("partition not covering model");}
};
inline Plan parse_plan(const std::string& spec,uint32_t layers,Geometry geometry,uint64_t fp) {
 Plan p;p.total_layers=layers;p.geometry=geometry;p.model_fingerprint=fp;
 std::istringstream in(spec);std::string item;
 while(std::getline(in,item,',')) {
  Stage s;char dash=0;const size_t a=item.find(':'),b=item.find(':',a+1),c=item.find('-',b+1);
  if(a==std::string::npos||b==std::string::npos||c==std::string::npos)throw std::invalid_argument("bad stage");
  const auto backend=item.substr(0,a);s.backend=backend=="cuda"?Backend::cuda:backend=="hip"?Backend::hip:throw std::invalid_argument("bad backend");
  s.ordinal=std::stoul(item.substr(a+1,b-a-1));s.first_layer=std::stoul(item.substr(b+1,c-b-1));s.end_layer=std::stoul(item.substr(c+1));p.stages.push_back(s);(void)dash;
 }p.validate();return p;
}
}
