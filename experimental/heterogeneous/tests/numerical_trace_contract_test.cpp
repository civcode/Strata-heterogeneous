#include "strata/core/numerical_trace.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
namespace nt=strata::core::numerical_trace;
#define REQUIRE(x) do {if(!(x))throw std::runtime_error(#x);}while(0)
struct Scratch {
    std::filesystem::path path;
    Scratch(){char root[]="/tmp/strata-numerical-contract-XXXXXX";auto* p=::mkdtemp(root);
        if(!p)throw std::runtime_error("mkdtemp");
        path=p;}
    ~Scratch(){std::error_code ec;std::filesystem::remove_all(path,ec);}
};
int main(){try {
    for(int c:{2,6})for(unsigned seq=0;seq<20;++seq)for(int p=-1;p<8;++p){
        const int a=c+1;int r=0;
        const int want=seq<unsigned(a)?int(seq):seq==unsigned(a)?c:int(seq)-a-1;
        const bool valid=seq<unsigned(2*a+1)&&p==want;
        REQUIRE(nt::identity(seq,c,p,p>=0&&p<=6?nt::tokens[p]:-1,r)==valid);
        REQUIRE(!nt::identity(seq,c,p,-1,r));
        if(valid){REQUIRE(nt::identity(seq,c,p,nt::tokens[p],r));REQUIRE(r==(seq<unsigned(a)?1:seq==unsigned(a)?2:3));}
    }
    int r=0;REQUIRE(!nt::identity(0,1,0,nt::tokens[0],r));
    REQUIRE(nt::stage(0,12)==0&&nt::stage(12,30)==1&&nt::stage(30,48)==2&&nt::stage(12,48)==-1);
    size_t end=0;for(int i=0;i<nt::count;++i){REQUIRE(nt::offset(static_cast<nt::Field>(i))==end);end+=nt::fields[size_t(i)].words;}
    REQUIRE(end==nt::stride && end==4*nt::N+2*nt::D+nt::NE+2*nt::K+nt::HC);
    Scratch temp;std::string e;const std::string path=(temp.path/"sample").string();
    const float data[]={1.f,-2.f};
    REQUIRE(nt::write(path,data,sizeof data,e));
    REQUIRE(!nt::write(path,data,sizeof data,e));
    struct stat info{};REQUIRE(::stat(path.c_str(),&info)==0&&(info.st_mode&0777)==0600);
    float got[2]={};std::ifstream f(path,std::ios::binary);REQUIRE(bool(f.read(reinterpret_cast<char*>(got),sizeof got)));
    REQUIRE(std::memcmp(got,data,sizeof data)==0);
    std::filesystem::create_symlink(path,temp.path/"alias");
    REQUIRE(!nt::write((temp.path/"alias").string(),data,sizeof data,e));
    REQUIRE(!nt::write((temp.path/"empty").string(),data,0,e));
    REQUIRE(!nt::write((temp.path/"bad").string(),nullptr,sizeof data,e));
    REQUIRE(!nt::write((temp.path/"large").string(),data,(16u<<20)+1,e));
    REQUIRE(nt::stem("/tmp/root",2,47,6)=="/tmp/root.trace.r2.l47.p6.");
    REQUIRE(nt::weight("/tmp/root",false,47)=="/tmp/root.trace.serial.l47.router.bf16");
    REQUIRE(nt::state("/tmp/root",3,2,6)=="/tmp/root.trace.r3.s2.p6.state.");
    std::cout<<"PASS: bounded sequence, field layout and exclusive numerical trace output\n";
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
