#pragma once
// Host-only diagnostic files: never a production transport.
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace strata::heterogeneous {
inline constexpr size_t prefill_stage_row_floats=4u*2560u;
inline constexpr size_t prefill_stage_max_rows=16;
inline std::string prefill_stage_file(const std::string& prefix,size_t stage,int64_t pos,const char* direction) {
    return prefix+".native.stage"+std::to_string(stage)+".p"+std::to_string(pos)+"."+direction+".f32";
}
inline bool prefill_stage_shape(size_t stage,int64_t pos,int64_t rows,size_t width) {
    return stage<3 && pos>=0 && rows>0 && rows<=int64_t(prefill_stage_max_rows) &&
        pos<=int64_t(prefill_stage_max_rows)-rows && width==prefill_stage_row_floats;
}
inline bool prefill_stage_finite(const float* p,size_t count) {
    if (!p || !count || count>prefill_stage_row_floats*prefill_stage_max_rows) return false;
    for(size_t i=0;i<count;++i) if(!std::isfinite(p[i])) return false;
    return true;
}
inline bool prefill_stage_write(const std::string& path,const float* data,size_t count,std::string& error) {
    if(!prefill_stage_finite(data,count)){error="stage sample nonfinite or invalid extent";return false;}
    const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
    if(fd<0){error="stage sample exclusive open failed: "+path;return false;}
    const auto* p=reinterpret_cast<const uint8_t*>(data);
    const size_t bytes=count*sizeof(float);
    size_t at=0;bool ok=true;
    while(at<bytes){const ssize_t n=::write(fd,p+at,bytes-at);
        if(n>0)at+=size_t(n);else if(n<0 && errno==EINTR)continue;else {ok=false;break;}}
    if(::close(fd)!=0)ok=false;
    if(!ok){::unlink(path.c_str());error="stage sample write failed: "+path;}
    return ok;
}
inline bool prefill_stage_read(const std::string& path,float* data,size_t count,std::string& error) {
    if(!data || !count || count>prefill_stage_row_floats*prefill_stage_max_rows){error="invalid replay extent";return false;}
    const int fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK);
    if(fd<0){error="stage replay open failed: "+path;return false;}
    const size_t bytes=count*sizeof(float);struct stat st{};
    bool ok=::fstat(fd,&st)==0 && S_ISREG(st.st_mode) && st.st_size>=0 && uint64_t(st.st_size)==bytes;
    auto* p=reinterpret_cast<uint8_t*>(data);size_t at=0;
    while(ok && at<bytes){const ssize_t n=::read(fd,p+at,bytes-at);
        if(n>0)at+=size_t(n);else if(n<0 && errno==EINTR)continue;else ok=false;}
    uint8_t extra=0;ssize_t n=-1;
    if(ok){do{n=::read(fd,&extra,1);}while(n<0&&errno==EINTR);ok=n==0;}
    if(::close(fd)!=0)ok=false;
    if(!ok || !prefill_stage_finite(data,count)){error="invalid stage replay file: "+path;return false;}
    return true;
}
} // namespace strata::heterogeneous
