#pragma once
#include "strata/core/weights.hpp"
#include <set>
#include <string>
#include <vector>
#include <cstdint>
namespace strata::core {
namespace fake_native {
inline bool no_qkv=false,fail_price=false,fail_load=false,fill_key=true;
inline bool bind_qkv=true,bad_key=false;
inline int opens=0,loads=0,destruct=0;
inline void reset(){no_qkv=fail_price=fail_load=false;fill_key=true;
    bind_qkv=true;bad_key=false;
    opens=loads=destruct=0;}
}
class NativeDense {
public:
    ~NativeDense(){++fake_native::destruct;}
    static bool served_names(const std::vector<std::string>& shards,bool,
                             std::set<std::string>& out,std::string& err){
        ++fake_native::opens;
        if(shards.empty()){err="no GGUF";return false;}
        if(!fake_native::no_qkv)out.insert("blk.0.attn_qkv.weight");
        out.insert("blk.1.ple_key.weight");
        return true;
    }
    static bool weight_bytes_for(const std::vector<std::string>&,
                WeightTable&,bool,int64_t,int64_t,uint64_t& out,std::string& err){
        if(fake_native::fail_price){err="native price failed";return false;}
        out=4096;return true;
    }
    bool load(const std::vector<std::string>&,WeightTable& wt,std::string& err,
              bool,int64_t,int64_t){
        ++fake_native::loads;
        if(fake_native::fail_load){err="native load failed";return false;}
        auto& entries=wt.modifiable();
        if(fake_native::bind_qkv)
            entries["blk.0.attn_qkv.weight"].native_data=reinterpret_cast<const void*>(0x1000);
        if(fake_native::fill_key){
            auto& key=entries["blk.1.ple_key.weight"];
            key.native_data=reinterpret_cast<const void*>(0x2000);
            key.native_q8_1=reinterpret_cast<void*>(0x3000);
            key.native_type=fake_native::bad_key ? 99 : 42;
        }
        return true;
    }
};
}
