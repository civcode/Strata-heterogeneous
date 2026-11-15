#include "strata_heterogeneous/prefill_stage_files.hpp"
#include "strata_heterogeneous/mtp_prefix_cursor.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>
using namespace strata::heterogeneous;
#define REQUIRE(x) do {if (!(x)) throw std::runtime_error(#x);} while (0)
struct Scratch {
    std::filesystem::path path;
    Scratch(){char root[]="/tmp/strata-stage-contract-XXXXXX";auto* p=::mkdtemp(root);
        if (!p) throw std::runtime_error("mkdtemp");
        path=p;}
    ~Scratch(){std::error_code ec;std::filesystem::remove_all(path,ec);}
};
int main(){
    try {
        Scratch tmp;
        std::string error;auto p=prefill_stage_file((tmp.path/"cap").string(),0,0,"out");
        REQUIRE(p==(tmp.path/"cap.native.stage0.p0.out.f32").string());
        for(size_t s=0;s<4;++s)for(int pos=-1;pos<=17;++pos)for(int n=-1;n<=17;++n){
            REQUIRE(prefill_stage_shape(s,pos,n,10240)==(s<3&&pos>=0&&n>0&&pos+n<=16));
            REQUIRE(!prefill_stage_shape(s,pos,n,10239));
        }
        std::vector<float> a(10240),b(10240);for(size_t i=0;i<a.size();++i)a[i]=float(i)/16;
        REQUIRE(prefill_stage_write(p,a.data(),a.size(),error));
        REQUIRE(!prefill_stage_write(p,a.data(),a.size(),error));
        REQUIRE(prefill_stage_read(p,b.data(),b.size(),error)&&a==b);
        REQUIRE(!prefill_stage_read(p,b.data(),b.size()-1,error));
        REQUIRE(!prefill_stage_read(p,nullptr,b.size(),error));
        std::filesystem::create_symlink(p,tmp.path/"alias");
        REQUIRE(!prefill_stage_read((tmp.path/"alias").string(),b.data(),b.size(),error));
        REQUIRE(!prefill_stage_read(tmp.path.string(),b.data(),b.size(),error));
        a[0]=std::numeric_limits<float>::quiet_NaN();
        REQUIRE(!prefill_stage_write((tmp.path/"nan").string(),a.data(),a.size(),error));
        REQUIRE(!std::filesystem::exists(tmp.path/"nan"));
        REQUIRE(!prefill_stage_read((tmp.path/"missing").string(),b.data(),b.size(),error));
        using Pair=std::tuple<float,int32_t,int64_t>;
        std::vector<Pair> seen;
        auto prime=[&](const float* r,const int32_t* ids,size_t n,int64_t pos,std::string&){
            for(size_t i=0;i<n;++i)seen.emplace_back(r[2*i],ids[i],pos+int64_t(i));
            return true;
        };
        unsigned shapes=0;
        for (size_t n=2;n<=7;++n) for(size_t chunk=1;chunk<n;++chunk) {
            ++shapes;seen.clear();error.clear();MtpPrefixCursor cur(2,2048,248320,8);
            std::vector<int32_t> prompt;for(size_t i=0;i<n;++i)prompt.push_back(100+int32_t(i));
            std::array<float,16> r{};
            for(size_t pos=0;pos<n-1;){
                size_t rows=std::min(chunk,n-1-pos);
                for(size_t i=0;i<rows;++i)r[2*i]=float(pos+i+1);
                REQUIRE(cur.seed_bulk_prompt_chunk(prompt,int64_t(pos),r.data(),rows,prime,error));
                pos+=rows;REQUIRE(cur.committed_position()==int64_t(pos));r.fill(-99);
            }
            REQUIRE(seen.size()==n-1);
            for(size_t i=0;i<n-1;++i) REQUIRE(seen[i]==Pair(float(i+1),prompt[i+1],int64_t(i)));
            REQUIRE(cur.begin(int64_t(n-1),{prompt.back()},prime,error)); // must not prime a stale tail
            REQUIRE(seen.size()==n-1);r[0]=float(n);
            REQUIRE(cur.committed(r.data(),1,prime,error));
            REQUIRE(cur.finish(42,[&](const float* tail,int32_t tok,int64_t pos,std::string&){
                return tail[0]==float(n)&&tok==42&&pos==int64_t(n-1);},error));
        }
        REQUIRE(shapes==21);
        // Refuse malformed seeds, premature/adversarial last prompt and reuse.
        std::array<float,8> r{};
        for(int fault=0;fault<12;++fault){
            MtpPrefixCursor cur(2,2048,248320,8);error.clear();seen.clear();
            const std::vector<int32_t> ids{11,12,13};bool refused=false;
            if(fault==0)refused=!cur.seed_bulk_prompt_chunk(ids,1,r.data(),1,prime,error);
            if(fault==1)refused=!cur.seed_bulk_prompt_chunk(ids,0,r.data(),3,prime,error);
            if(fault==2)refused=!cur.seed_bulk_prompt_chunk({11,248320},0,r.data(),1,prime,error);
            if(fault==3)refused=!cur.seed_bulk_prompt_chunk(ids,0,nullptr,1,prime,error);
            if(fault==4)refused=!cur.seed_bulk_prompt_chunk(std::vector<int32_t>(2048,1),0,r.data(),1,prime,error);
            if(fault==5)refused=!cur.seed_bulk_prompt_chunk(ids,0,r.data(),1,[](auto...){return false;},error);
            if(fault>=6){
                REQUIRE(cur.seed_bulk_prompt_chunk(ids,0,r.data(),1,prime,error));
                if(fault==6)refused=!cur.seed_bulk_prompt_chunk({11,12,14},1,r.data(),1,prime,error);
                if(fault==7)refused=!cur.seed_bulk_prompt_chunk(ids,0,r.data(),1,prime,error);
                if(fault==8)refused=!cur.begin(1,{12},prime,error);
                if(fault>=9){
                    REQUIRE(cur.seed_bulk_prompt_chunk(ids,1,r.data(),1,prime,error));
                    if(fault==9)refused=!cur.begin(2,{14},prime,error);
                    if(fault==10)refused=!cur.seed_bulk_prompt_chunk(ids,2,r.data(),1,prime,error);
                    if(fault==11){REQUIRE(cur.begin(2,{13},prime,error));REQUIRE(cur.committed(r.data(),1,prime,error));
                        refused=!cur.seed_bulk_prompt_chunk(ids,0,r.data(),1,prime,error);}
                }
            }
            REQUIRE(refused && !error.empty());
            REQUIRE(!cur.begin(cur.committed_position(),{13},prime,error));
        }
        std::cout<<"PASS: stage file contract and 21 bulk MTP seed shapes; no GPU inference\n";return 0;
    } catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}
}
