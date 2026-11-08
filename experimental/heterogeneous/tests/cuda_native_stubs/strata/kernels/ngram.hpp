#pragma once
#include <cstdint>
#include <string>
namespace strata::kernels {
struct PleConsts {uint32_t flag=0;};
inline PleConsts ple_artifact_consts(){return PleConsts{37};}
enum class PleIo {Direct=0, Mmap=1};
struct PleIoOptions {PleIo mode=PleIo::Direct; uint64_t cache_rows=123; double keepalive_ms=99;};
class PleTable {
    bool open_=false;
public:
    PleTable()=default;~PleTable()=default;
    bool open(const std::string& path,std::string& err,const PleIoOptions&) {
        if(path.find("bad-ple")!=std::string::npos){err="bad PLE shard";return false;}
        open_=true;return true;
    }
    void close(){open_=false;}
    bool is_open() const {return open_;}
};
}
