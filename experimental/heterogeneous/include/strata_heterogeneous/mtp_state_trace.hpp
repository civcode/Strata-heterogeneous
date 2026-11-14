#pragma once

// Bounded read-only native draft trace. Raw exact bytes, not logits as a
// surrogate for residual/KV identity. Only committed-prefix cells are copied;
// conversation_kv_save internally rounds to pages, so trim unused suffix.
#include "strata/core/conversation_snapshot.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace strata::heterogeneous {
class MtpStateTrace {
public:
    static constexpr int64_t max_cells=16;
    MtpStateTrace(std::string prefix,const core::ModelGeometry& g)
        : prefix_(std::move(prefix)),g_(g) {}
    bool capture(const char* phase,const core::QsaState& state,const float* rows,
                 const int32_t* tokens,size_t n,int64_t cell,int64_t upto,std::string& error) {
        if(prefix_.empty()) return true;
        if (state.kv_mode!=0 || !rows || !tokens || !n || n>8 || cell<0 || cell>max_cells ||
            cell+int64_t(n)>max_cells || upto<0 || upto>cell+int64_t(n) ||
            upto>max_cells || events_>=64 || g_.hc!=4 || g_.n_embd!=2560 ||
            std::getenv("STRATA_PARK_ASYNC") || std::getenv("STRATA_PARK_VRAM_MIB")) {
            error="MTP state trace requires bounded fully resident draft KV and synchronous snapshot";return false;
        }
        if(cudaDeviceSynchronize()!=cudaSuccess) {error="MTP state trace drain failed";return false;}
        const std::string stem=prefix_+".mtp."+std::to_string(++events_);
        const size_t stride=size_t(g_.hc)*size_t(g_.n_embd)*sizeof(float);
        std::vector<uint8_t> residual(n*stride);
        if(cudaMemcpy(residual.data(),rows,residual.size(),cudaMemcpyDeviceToHost)!=cudaSuccess) {
            error="MTP state trace residual readback failed";return false;
        }
        core::ConversationKv kv;
        if(upto) {
            std::array<uint64_t,5> sizes{};
            if(!core::conversation_kv_part_sizes(state,g_,upto,false,sizes,error)) return false;
            uint64_t total=0;
            for(auto bytes:sizes) {
                if(bytes>(16ull<<20)-total) {error="MTP state trace KV cap exceeded";return false;}
                total+=bytes;
            }
            if(!core::conversation_kv_save(kv,state,g_,upto,false,error)) return false;
            // Native dense BF16 pools are [page][kv_head][cell][head_dim],
            // not cell-major. Reject other formats until explicitly decoded.
            if(kv.cells<upto || kv.page_size<=0 || kv.format!=0 || kv.pending || kv.dev_tag ||
               kv.pooled.size() || kv.heads<=0 || kv.head_dim<=0) {
                error="MTP state trace unexpected native KV layout";return false;
            }
        }
        if(!file(stem+".r.bin",[&](int fd){return write_all(fd,residual.data(),residual.size(),error);},error))return false;
        std::array<size_t,4> strides{};
        const std::array<const core::ConversationBuffer*,4> parts={&kv.k,&kv.v,&kv.k_scale,&kv.v_scale};
        const std::array<const char*,4> names={"k","v","ks","vs"};
        for(size_t i=0;i<parts.size();++i) {
            const auto& part=*parts[i];
            const size_t head_bytes=upto?size_t(kv.head_dim)*sizeof(uint16_t):0;
            strides[i]=(upto && i<2)?size_t(kv.heads)*head_bytes:0;
            if(upto && ((i<2 && part.size()!=size_t(kv.cells)*strides[i]) ||
                        (i>=2 && part.size()!=0))) {
                error="MTP state trace unexpected dense KV extent";return false;
            }
            if(!file(stem+"."+names[i]+".bin",[&](int fd){
                if(i>=2) return true;
                for(int64_t c=0;c<upto;++c) for(int64_t h=0;h<kv.heads;++h) {
                    const size_t row=(size_t(c/kv.page_size)*size_t(kv.heads)+size_t(h))*
                                     size_t(kv.page_size)+size_t(c%kv.page_size);
                    if(!part.visit(row*head_bytes,head_bytes,[&](const uint8_t* data,size_t size,size_t){
                        return write_all(fd,data,size,error);
                    })) return false;
                }
                return true;
            },error))return false;
        }
        // Manifest LAST; incomplete captures must never be compared.
        std::ostringstream meta;
        meta<<"{\"schema\":\"strata-mtp-state/v2\",\"event\":"<<events_
            <<",\"phase\":\""<<phase<<"\",\"cell\":"<<cell<<",\"upto\":"<<upto
            <<",\"residual_stride\":"<<stride<<",\"format\":"<<kv.format
            <<",\"kv_strides\":["<<strides[0]<<','<<strides[1]<<','<<strides[2]<<','<<strides[3]
            <<"],\"tokens\":[";
        for(size_t i=0;i<n;++i){if(i)meta<<',';meta<<tokens[i];}
        meta<<"]}\n";
        const auto body=meta.str();
        return file(stem+".json",[&](int fd){return write_all(fd,body.data(),body.size(),error);},error);
    }
private:
    std::string prefix_;
    const core::ModelGeometry& g_;
    unsigned events_=0;
    static bool write_all(int fd,const void* src,size_t n,std::string& error) {
        const auto* p=static_cast<const uint8_t*>(src);
        while(n) {
            const auto w=::write(fd,p,n);
            if(w<0 && errno==EINTR)continue;
            if(w<=0){error="MTP state trace write failed";return false;}
            p+=w;n-=size_t(w);
        }
        return true;
    }
    template<class Write>
    static bool file(const std::string& path,Write&& put,std::string& error) {
        const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
        if(fd<0){error=std::string("MTP state trace exclusive open: ")+std::strerror(errno);return false;}
        bool ok=false;
        try{ok=put(fd);}catch(...){(void)::close(fd);(void)::unlink(path.c_str());throw;}
        if(::close(fd)!=0){error="MTP state trace close failed";ok=false;}
        if(!ok)(void)::unlink(path.c_str());
        return ok;
    }
};
} // namespace strata::heterogeneous
