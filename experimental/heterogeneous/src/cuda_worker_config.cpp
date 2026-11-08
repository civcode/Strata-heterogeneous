#include "strata_heterogeneous/cuda_worker_config.hpp"

#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace strata::heterogeneous {
namespace {
// Small dependency-free SHA-256: index.txt is bounded to 8 MiB and the
// digest is a preflight consistency check, not a signature/attestation.
constexpr uint32_t k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
class Sha256 {
    std::array<uint32_t,8> h_{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                               0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
    std::array<uint8_t,64> block_{};
    size_t buffered_ = 0;
    uint64_t total_ = 0;
    void compress() {
        uint32_t w[64]{};
        for (size_t i=0;i<16;++i) w[i] = (uint32_t(block_[i*4])<<24) |
            (uint32_t(block_[i*4+1])<<16) | (uint32_t(block_[i*4+2])<<8) | block_[i*4+3];
        for (size_t i=16;i<64;++i) {
            const uint32_t s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
            const uint32_t s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h_[0],b=h_[1],c=h_[2],d=h_[3],e=h_[4],f=h_[5],g=h_[6],z=h_[7];
        for(size_t i=0;i<64;++i){
            const uint32_t s1=rotr(e,6)^rotr(e,11)^rotr(e,25);
            const uint32_t ch=(e&f)^(~e&g);
            const uint32_t t1=z+s1+ch+k[i]+w[i];
            const uint32_t s0=rotr(a,2)^rotr(a,13)^rotr(a,22);
            const uint32_t maj=(a&b)^(a&c)^(b&c);
            const uint32_t t2=s0+maj;
            z=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h_[0]+=a;h_[1]+=b;h_[2]+=c;h_[3]+=d;h_[4]+=e;h_[5]+=f;h_[6]+=g;h_[7]+=z;
    }
public:
    void append(const uint8_t* data,size_t n) {
        total_+=n;
        for(size_t i=0;i<n;++i){block_[buffered_++]=data[i];if(buffered_==64){compress();buffered_=0;}}
    }
    std::string finish() {
        const uint64_t bits=total_*8;
        const uint8_t one=0x80,zero=0;
        append(&one,1);
        while(buffered_!=56)append(&zero,1);
        uint8_t len[8];for(unsigned i=0;i<8;++i)len[7-i]=static_cast<uint8_t>(bits>>(i*8));
        append(len,8);
        constexpr char hexdigits[]="0123456789abcdef";
        std::string hex;hex.reserve(64);
        for(uint32_t word:h_)for(int shift=24;shift>=0;shift-=8){
            const uint8_t byte=static_cast<uint8_t>(word>>shift);
            hex.push_back(hexdigits[byte>>4]);hex.push_back(hexdigits[byte&15]);
        }
        return hex;
    }
};

template<class T>T decimal(const std::string& s,const std::string& key) {
    if(s.empty()||s.front()=='+'||s.front()=='-'||s.front()==' ')throw std::invalid_argument("invalid "+key);
    T n{}; const auto r=std::from_chars(s.data(),s.data()+s.size(),n);
    if(r.ec!=std::errc{}||r.ptr!=s.data()+s.size())throw std::invalid_argument("invalid "+key);
    return n;
}
void absolute_path(const std::string& path,const char* key) {
    if(path.size()<2||path.size()>4096||path[0]!='/'||path.find('\0')!=std::string::npos||
       path.find('\r')!=std::string::npos||path.find('\n')!=std::string::npos)
        throw std::invalid_argument(std::string(key)+" must be an absolute local path");
    const std::filesystem::path p(path);
    for(const auto& component:p)if(component==".."||component==".")
        throw std::invalid_argument(std::string(key)+" must not contain dot path segments");
}
void regular_file(const std::string& path,const char* key) {
    std::error_code ec;
    if(!std::filesystem::is_regular_file(path,ec)||ec)
        throw std::runtime_error(std::string(key)+" is not a readable regular file: "+path);
}
std::string file_digest(const std::string& path) {
    regular_file(path,"index.txt");
    std::error_code ec;
    const auto size=std::filesystem::file_size(path,ec);
    if(ec||size==0||size>8ull*1024*1024)throw std::runtime_error("pack index size is invalid (max 8 MiB)");
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("cannot open pack index: "+path);
    Sha256 sha;
    std::array<uint8_t,8192> buf{};
    while(in){
        in.read(reinterpret_cast<char*>(buf.data()),static_cast<std::streamsize>(buf.size()));
        const std::streamsize n=in.gcount();
        if(n>0)sha.append(buf.data(),static_cast<size_t>(n));
    }
    if(!in.eof())throw std::runtime_error("error reading pack index");
    return sha.finish();
}
const std::set<std::string>& keys() {
    static const std::set<std::string> values={
        "version","pack_dir","profile_path","index_sha256","model_fingerprint",
        "stage_index","device_ordinal","first_layer","end_layer","n_layers","hc",
        "n_embd","n_vocab","max_cells","max_t","cpu_workers","max_cache_slots",
        "io_timeout_ms","free_reserve_bytes","arena_budget_bytes"};
    return values;
}
} // namespace

bool read_cuda_worker_config(const std::string& filename,CudaWorkerConfig& out,std::string& error) {
    try {
        absolute_path(filename,"config path");
        regular_file(filename,"config");
        std::error_code ec;
        const auto size=std::filesystem::file_size(filename,ec);
        if(ec||size>16384||size==0)throw std::runtime_error("config must be nonempty and at most 16 KiB");
        std::ifstream in(filename);
        if(!in)throw std::runtime_error("cannot open worker config");
        std::map<std::string,std::string> fields;
        std::string line;
        while(std::getline(in,line)){
            if(line.empty()||line[0]=='#')continue;
            if(line.size()>4096)throw std::invalid_argument("config line too long");
            const auto eq=line.find('=');
            if(eq==std::string::npos||eq==0||eq==line.size()-1||line.find('=',eq+1)!=std::string::npos)
                throw std::invalid_argument("config requires one key=value per line");
            const auto key=line.substr(0,eq),value=line.substr(eq+1);
            if(!keys().count(key)||!fields.emplace(key,value).second)
                throw std::invalid_argument("unknown or duplicate CUDA config key: "+key);
        }
        if(!in.eof()||fields.size()!=keys().size())throw std::invalid_argument("config lacks required fields or has read error");
        const auto& v=[&](const char* key)->const std::string& {return fields.at(key);};
        if(v("version")!="1")throw std::invalid_argument("unsupported CUDA worker config version");
        CudaWorkerConfig c;
        c.pack_dir=v("pack_dir");c.profile_path=v("profile_path");
        absolute_path(c.pack_dir,"pack_dir");absolute_path(c.profile_path,"profile_path");
        c.index_sha256=v("index_sha256");
        if(c.index_sha256.size()!=64||c.index_sha256.find_first_not_of("0123456789abcdef")!=std::string::npos)
            throw std::invalid_argument("index_sha256 must be 64 lowercase hex digits");
        c.model_fingerprint=decimal<uint64_t>(v("model_fingerprint"),"model_fingerprint");
        c.stage_index=decimal<uint32_t>(v("stage_index"),"stage_index");
        c.device_ordinal=decimal<uint32_t>(v("device_ordinal"),"device_ordinal");
        c.first_layer=decimal<uint32_t>(v("first_layer"),"first_layer");
        c.end_layer=decimal<uint32_t>(v("end_layer"),"end_layer");
        c.n_layers=decimal<uint32_t>(v("n_layers"),"n_layers");
        c.hc=decimal<uint32_t>(v("hc"),"hc");
        c.n_embd=decimal<uint32_t>(v("n_embd"),"n_embd");
        c.n_vocab=decimal<uint32_t>(v("n_vocab"),"n_vocab");
        c.max_cells=decimal<uint32_t>(v("max_cells"),"max_cells");
        c.max_t=decimal<uint32_t>(v("max_t"),"max_t");
        c.cpu_workers=decimal<uint32_t>(v("cpu_workers"),"cpu_workers");
        c.max_cache_slots=decimal<uint32_t>(v("max_cache_slots"),"max_cache_slots");
        c.io_timeout_ms=decimal<uint32_t>(v("io_timeout_ms"),"io_timeout_ms");
        c.free_reserve_bytes=decimal<uint64_t>(v("free_reserve_bytes"),"free_reserve_bytes");
        c.arena_budget_bytes=decimal<uint64_t>(v("arena_budget_bytes"),"arena_budget_bytes");
        if(!c.model_fingerprint||!c.n_layers||!c.hc||!c.n_embd||!c.n_vocab||
           c.first_layer>=c.end_layer||c.end_layer>c.n_layers||
           c.max_cells<2||c.max_cells>1'000'000||c.max_t<2||c.max_t>8||
           c.cpu_workers<1||c.cpu_workers>28||c.max_cache_slots<1||c.max_cache_slots>1024||
           c.io_timeout_ms<100||c.io_timeout_ms>120000||c.free_reserve_bytes<(128ull<<20))
            throw std::invalid_argument("CUDA config geometry or bounded execution options are invalid");
        out=std::move(c);error.clear();return true;
    }catch(const std::exception& ex){error=ex.what();return false;}
}

bool validate_cuda_worker_files(const CudaWorkerConfig& c,std::string& error) {
    try {
        absolute_path(c.pack_dir,"pack_dir");absolute_path(c.profile_path,"profile_path");
        std::error_code ec;
        if(!std::filesystem::is_directory(c.pack_dir,ec)||ec)
            throw std::runtime_error("pack_dir is not a readable directory");
        const auto digest=file_digest(c.pack_dir+"/index.txt");
        if(digest!=c.index_sha256)throw std::runtime_error("pack index SHA-256 does not match trusted config");
        regular_file(c.profile_path,"profile_path");
        std::ifstream profile(c.profile_path,std::ios::binary);
        char prefix[8]{};profile.read(prefix,8);
        if(profile.gcount()!=8||std::string(prefix,4)!="STRP"||
           static_cast<uint8_t>(prefix[4])!=1||prefix[5]!=0||prefix[6]!=0||prefix[7]!=0)
            throw std::runtime_error("expert profile is not STRP/v1");
        error.clear();return true;
    }catch(const std::exception& ex){error=ex.what();return false;}
}

bool validate_cuda_worker_config(const CudaWorkerConfig& c,const Plan& plan,
                                 size_t index,std::string& error) {
    try {
        plan.validate();
        if(index>=plan.stages.size()||index!=c.stage_index)throw std::invalid_argument("CUDA stage index mismatch");
        const Stage& s=plan.stages[index];
        if(s.backend!=Backend::cuda||s.ordinal!=c.device_ordinal||
           s.first_layer!=c.first_layer||s.end_layer!=c.end_layer||
           plan.total_layers!=c.n_layers||plan.geometry.hc!=c.hc||
           plan.geometry.n_embd!=c.n_embd||plan.model_fingerprint!=c.model_fingerprint)
            throw std::invalid_argument("CUDA config and coordinator plan identity mismatch");
        if(!validate_cuda_worker_files(c,error))return false;
        return true;
    }catch(const std::exception& ex){error=ex.what();return false;}
}
} // namespace strata::heterogeneous
