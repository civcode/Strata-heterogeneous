// CUDA-only native HC operator replay, not an inference engine or benchmark.
// Inputs are bounded read-only actual layer-zero capture files. No model loader,
// heterogeneous coordinator, HIP runtime, remote memory, sockets or transport.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(STRATA_USE_HIP) || defined(STRATA_GPU_METAL)
#error "CUDA-only native HC replay"
#endif
namespace fs=std::filesystem;
namespace k=strata::kernels;
namespace pf=strata::prefill;
namespace {
constexpr size_t N=2560, HC=4, D=N*HC, LR=320, Q=N/32*36;
constexpr float EPS=1e-6f;
void ck(cudaError_t e,const char* what) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));
}
template<class T> std::vector<T> read(const fs::path& path,size_t n) {
    if(!fs::is_regular_file(path) || fs::file_size(path)!=n*sizeof(T))
        throw std::runtime_error("invalid input extent: "+path.string());
    std::vector<T> v(n);
    std::ifstream f(path,std::ios::binary);
    if(!f.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(T)))
        throw std::runtime_error("input read failed: "+path.string());
    return v;
}
template<class T> void write(const fs::path& path,const std::vector<T>& v) {
    if(fs::exists(path)) throw std::runtime_error("output exists: "+path.string());
    std::ofstream f(path,std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));f.close();
    if(!f) throw std::runtime_error("output write failed: "+path.string());
}
struct Stream {
    cudaStream_t value=nullptr;
    Stream(){ck(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking),"create stream");}
    ~Stream(){if(value){if(cudaStreamSynchronize(value)!=cudaSuccess || cudaStreamDestroy(value)!=cudaSuccess)std::_Exit(70);}}
};
template<class T> struct Dev {
    T* p=nullptr; size_t n;
    explicit Dev(size_t count):n(count) {ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)),"alloc");}
    Dev(const Dev&)=delete;Dev& operator=(const Dev&)=delete;
    ~Dev(){if(p && (cudaDeviceSynchronize()!=cudaSuccess || cudaFree(p)!=cudaSuccess))std::_Exit(70);}
    void upload(const std::vector<T>& v) {
        if(v.size()!=n)throw std::runtime_error("upload size");
        ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice),"upload");
        // Nonblocking native compute streams do not inherit default-stream
        // readiness. Fence this diagnostic's uploads before any replay.
        ck(cudaDeviceSynchronize(),"upload fence");
    }
    void zero(){ck(cudaMemset(p,0,n*sizeof(T)),"zero");ck(cudaDeviceSynchronize(),"zero fence");}
    std::vector<T> get(cudaStream_t s) {
        ck(cudaStreamSynchronize(s),"readback fence");
        std::vector<T> v(n);
        ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost),"readback");
        return v;
    }
};
template<class T> void export_dev(const fs::path& out,const char* name,Dev<T>& d,cudaStream_t s) {
    write(out/name,d.get(s));
}
void finite(const std::vector<float>& x) {
    for(float v:x)if(!std::isfinite(v))throw std::runtime_error("nonfinite FP32 input/output");
}
void bf_finite(const std::vector<uint16_t>& x) {
    for(uint16_t v:x)if((v&0x7f80u)==0x7f80u)throw std::runtime_error("nonfinite BF16 input/output");
}
struct Input {
    std::vector<float> residual,norm;
    std::vector<uint16_t> down,up,inject;
    Dev<float> r{D},w{D};
    Dev<uint16_t> wd{D*LR},wu{D*LR},wi{D*HC};
    explicit Input(const std::string& prefix):
      residual(read<float>(prefix+".residual.f32",D)),
      norm(read<float>(prefix+".norm_weight.f32",D)),
      down(read<uint16_t>(prefix+".down_weight.bf16",D*LR)),
      up(read<uint16_t>(prefix+".up_weight.bf16",D*LR)),
      inject(read<uint16_t>(prefix+".inject_weight.bf16",D*HC)) {
        finite(residual);finite(norm);bf_finite(down);bf_finite(up);bf_finite(inject);
        r.upload(residual);w.upload(norm);wd.upload(down);wu.upload(up);wi.upload(inject);
    }
};
void serial(Input& in,const fs::path& out,bool qfuse,bool& wrote_q8) {
    Stream st;
    Dev<float> residual(D),scratch(D),lo(LR),rs(HC),inj(HC),mixed(N);
    Dev<unsigned> counters(N/32);
    Dev<uint8_t> q8(Q);
    residual.upload(in.residual);
    scratch.zero();counters.zero();q8.zero();
    // Some fused variants keep the bottleneck in shared memory rather than
    // writing a.lo. Sentinel tells the reader that it is NOT OBSERVED.
    lo.upload(std::vector<float>(LR,std::numeric_limits<float>::quiet_NaN()));
    rs.upload(std::vector<float>(HC,std::numeric_limits<float>::quiet_NaN()));
    inj.upload(std::vector<float>(HC,std::numeric_limits<float>::quiet_NaN()));
    mixed.upload(std::vector<float>(N,std::numeric_limits<float>::quiet_NaN()));
    k::FusedGrArgs a;
    a.R=residual.p;a.R_out=residual.p;a.apply=false;
    a.w_norm=in.w.p;a.w_down=in.wd.p;a.w_up=in.wu.p;a.w_inject=in.wi.p;
    a.eps=EPS;a.lo=lo.p;a.rs=rs.p;a.inject_out=inj.p;a.mixed=mixed.p;
    if(qfuse){a.q8_mixed=q8.p;a.q8_cnt=counters.p;}
    if(k::fused_gr_read_persist(&a,1,st.value))
        throw std::runtime_error("unexpected persistent HC path on CUDA");
    wrote_q8=k::fused_gr_read_multi(&a,1,scratch.p,st.value);
    ck(cudaGetLastError(),"native fused HC launch");
    auto scales=rs.get(st.value), bottleneck=lo.get(st.value), injection=inj.get(st.value), output=mixed.get(st.value);
    finite(scales);finite(injection);finite(output);
    bool missing=true;for(float x:bottleneck)missing=missing && std::isnan(x);
    if(!missing){finite(bottleneck);write(out/"silu.f32",bottleneck);}
    write(out/"scale.f32",scales);write(out/"inject.f32",injection);write(out/"mixed.f32",output);
    if(wrote_q8)export_dev(out,"mixed.q8_1",q8,st.value);
    const auto readback=residual.get(st.value);
    if(std::memcmp(readback.data(),in.residual.data(),D*sizeof(float)))
        throw std::runtime_error("non-applying HC modified residual");
}
void native(Input& in,const fs::path& out) {
    Stream st;
    // Same owned workspace size and default math mode as Prefill::init.
    Dev<uint16_t> scratch(32u<<20);
    Dev<uint8_t> workspace(32u<<20);
    pf::Gemm gm;
    std::string error;
    if(!gm.init_external(st.value,scratch.p,scratch.n,workspace.p,workspace.n,error))
        throw std::runtime_error(error);
    if(pf::prompt_f16() || gm.bf16_converts_x())
        throw std::runtime_error("not plain CUDA BF16 GEMM path");
    gm.set_f16_io(false);pf::set_act_f16(false);
    Dev<float> rs(HC),down(LR),up(D),inj(HC),mixed(N);
    Dev<uint16_t> xn(D),silu(LR),mixed_bf(N),mixed_h(N);
    pf::gr_norm_rs(in.r.p,in.w.p,EPS,rs.p,xn.p,1,st.value);
    gm.bf16(xn.p,in.wd.p,down.p,1,LR,D);
    pf::gr_silu(down.p,silu.p,1,st.value);
    gm.bf16(silu.p,in.wu.p,up.p,1,D,LR);
    gm.bf16(xn.p,in.wi.p,inj.p,1,HC,D);
    pf::gr_mix_r(in.r.p,rs.p,in.w.p,up.p,mixed.p,mixed_bf.p,1,st.value,mixed_h.p);
    ck(cudaGetLastError(),"native prompt HC launch");
    export_dev(out,"scale.f32",rs,st.value);export_dev(out,"normalized.bf16",xn,st.value);
    export_dev(out,"down.f32",down,st.value);export_dev(out,"silu.bf16",silu,st.value);
    export_dev(out,"up.f32",up,st.value);export_dev(out,"inject.f32",inj,st.value);
    export_dev(out,"mixed.f32",mixed,st.value);
    const auto readback=in.r.get(st.value);
    if(std::memcmp(readback.data(),in.residual.data(),D*sizeof(float)))
        throw std::runtime_error("prompt HC modified residual");
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=5 || (std::string(argv[4])!="0" && std::string(argv[4])!="1"))
            throw std::runtime_error("usage: strata-hc-replay ROOT.hc NEW_OUTPUT_DIRECTORY CUDA_ORDINAL QFUSE_0_OR_1");
        const std::string ord=argv[3];size_t end=0;int device=std::stoi(ord,&end);
        if(end!=ord.size() || device<0 || device>63)throw std::runtime_error("invalid CUDA ordinal");
        const bool qfuse=std::string(argv[4])=="1";
        ck(cudaSetDevice(device),"set CUDA device");
        cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,device),"device properties");
        if(prop.major<8)throw std::runtime_error("plain BF16 replay requires sm80+");
        char pci[32]={};ck(cudaDeviceGetPCIBusId(pci,sizeof pci,device),"PCI bus ID");
        const fs::path out(argv[2]);
        if(!fs::create_directory(out))throw std::runtime_error("new output directory required");
        fs::permissions(out,fs::perms::owner_all,fs::perm_options::replace);
        for(const auto* mode:{"inputs","serial-a","native","serial-b"})
            if(!fs::create_directory(out/mode))throw std::runtime_error("new mode directory required");
        k::gr_set_native_mmvf(true);
        k::fused_gr_check();k::fused_gr_persist_init();
        bool qa=false,qb=false;
        {
            Input in(argv[1]);
            // Retain the actual host bytes consumed, not merely the input path.
            write(out/"inputs"/"residual.f32",in.residual);
            write(out/"inputs"/"norm_weight.f32",in.norm);
            write(out/"inputs"/"down_weight.bf16",in.down);
            write(out/"inputs"/"up_weight.bf16",in.up);
            write(out/"inputs"/"inject_weight.bf16",in.inject);
            serial(in,out/"serial-a",qfuse,qa);
            native(in,out/"native");
            serial(in,out/"serial-b",qfuse,qb);
        }
        if(qa!=qb)throw std::runtime_error("inconsistent fused Q8 publication");
        ck(cudaDeviceSynchronize(),"final drain");
        std::ofstream m(out/"complete.json");
        m<<"{\"schema\":\"strata-native-hc-replay/v1\",\"cuda_device\":"<<device
         <<",\"pci\":\""<<pci<<"\",\"qfuse\":"<<(qfuse?"true":"false")
         <<",\"q8_written\":"<<(qa?"true":"false")<<",\"variant\":"<<k::fused_gr_variant()
         <<",\"n_embd\":2560,\"hc\":4,\"rank\":320,\"rows\":1}\n";
        m.close();if(!m)throw std::runtime_error("completion marker failed");
        std::cout<<"HETERO_HC_REPLAY_CAPTURED native-only operator A/B/A; not qualification\n";
        return 0;
    } catch(const std::exception& ex) {
        std::cerr<<"HC replay refused: "<<ex.what()<<'\n';return 1;
    }
}
