#pragma once
// Bounded diagnostic-only native trace. No inference inputs are modified.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif
namespace strata::core::numerical_trace {
inline constexpr size_t N=2560,D=10240,NE=512,K=10,HC=4;
inline constexpr int32_t tokens[]={248045,846,198,2523,513,32767,264};
enum Field { r_attn,attn_mixed,attn_bo,r_ffn,ffn_mixed,router_logits,router_ids,router_weights,ffn_bo,ffn_inj,count };
struct Spec { const char* name; size_t words; };
inline constexpr std::array<Spec,count> fields={{{"r_attn.f32",D},{"attn_mixed.f32",N},{"attn_bo.f32",N},
    {"r_ffn.f32",D},{"ffn_mixed.f32",N},{"router_logits.f32",NE},{"router_ids.i32",K},
    {"router_weights.f32",K},{"ffn_bo.f32",N},{"ffn_inj.f32",HC}}};
inline constexpr size_t offset(Field f) {size_t n=0;for(int i=0;i<int(f);++i)n+=fields[size_t(i)].words;return n;}
inline constexpr size_t stride=offset(count);
inline const char* prefix() {
#if defined(__linux__) && !defined(STRATA_GPU_METAL)
    const char* p=std::getenv("STRATA_HETERO_NUMERICAL_TRACE_PREFIX");return p&&*p?p:nullptr;
#else
    return nullptr;
#endif
}
inline int cut() {
    const char* p=std::getenv("STRATA_HETERO_NUMERICAL_TRACE_CUT");
    if(p&&std::strlen(p)==1&&(p[0]=='2'||p[0]=='6'))return p[0]-'0';
    return 0;
}
inline int stage(int64_t lo,int64_t hi) {return lo==0&&hi==12?0:lo==12&&hi==30?1:lo==30&&hi==48?2:-1;}
inline bool identity(unsigned sequence,int c,int64_t pos,int32_t token,int& request) {
    if(c!=2&&c!=6)return false;
    const unsigned a=unsigned(c+1);
    if(sequence>=2*a+1)return false;
    int expected;
    if(sequence<a){request=1;expected=int(sequence);}
    else if(sequence==a){request=2;expected=c;}
    else {request=3;expected=int(sequence-a-1);}
    return pos==expected&&token==tokens[expected];
}
inline std::string stem(const std::string& root,int request,int64_t layer,int64_t pos) {
    return root+".trace.r"+std::to_string(request)+".l"+std::to_string(layer)+".p"+std::to_string(pos)+".";
}
inline std::string weight(const std::string& root,bool prompt,int64_t layer) {
    return root+".trace."+ (prompt?"native":"serial")+".l"+std::to_string(layer)+".router.bf16";
}
inline std::string state(const std::string& root,int request,int s,int c) {
    return root+".trace.r"+std::to_string(request)+".s"+std::to_string(s)+".p"+std::to_string(c)+".state.";
}
inline bool write(const std::string& path,const void* data,size_t bytes,std::string& err) {
    if(!data||!bytes||bytes>(16u<<20)){err="numerical trace invalid bytes";return false;}
#if defined(__linux__)
    const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
    if(fd<0){err="numerical trace exclusive create failed: "+path;return false;}
    const auto* p=static_cast<const uint8_t*>(data);size_t at=0;bool ok=true;
    while(at<bytes){const ssize_t n=::write(fd,p+at,bytes-at);
        if(n>0)at+=size_t(n);else if(n<0&&errno==EINTR)continue;else{ok=false;break;}}
    if(::close(fd)!=0)ok=false;
    if(!ok){::unlink(path.c_str());err="numerical trace write failed: "+path;}
    return ok;
#else
    (void)path;err="numerical trace unsupported platform";return false;
#endif
}
} // namespace strata::core::numerical_trace
