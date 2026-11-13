#include "strata_heterogeneous/hip_worker_config.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
using namespace strata::heterogeneous;
namespace {
int assertions = 0;
void need(bool ok, const char* why) {
    ++assertions;
    if (!ok) throw std::runtime_error(why);
}
struct Fixture {
    std::string root, pack, profile, gguf, cfg;
    Fixture() {
        char path[] = "/tmp/strata-m2-XXXXXX";
        if (!::mkdtemp(path)) throw std::runtime_error("mkdtemp failed");
        root = path; pack = root + "/pack"; profile = root + "/profile.strp";
        gguf = root + "/original.gguf"; cfg = root + "/hip.cfg";
        std::filesystem::create_directory(pack);
        {std::ofstream f(pack + "/index.txt");
         f << "# align 256 pool 512 tensors 1\n";
         f << "output.weight 0 0 0 10 0 512 2560 248320 0 0 0 0 0 0 0 0 0 0\n";}
        {std::ofstream f(profile, std::ios::binary);
         char hdr[24] = {}; hdr[0]='S';hdr[1]='T';hdr[2]='R';hdr[3]='P';hdr[4]=1;
         f.write(hdr, sizeof hdr);}
        {std::ofstream f(gguf,std::ios::binary); f << "GGUFfixture";}
    }
    ~Fixture() {std::error_code ec; std::filesystem::remove_all(root,ec);}
    std::string config(int stage=1,int begin=3,int end=7,bool ple=false) const {
        return "version=2\npack_dir="+pack+"\nprofile_path="+profile+
            "\nindex_sha256=7b7894ba64859946e9f6febcfbc1e2470696c076e9585a252d040c6ce96c9861"
            "\nmodel_fingerprint=999\nstage_index="+std::to_string(stage)+
            "\ndevice_ordinal=0\nfirst_layer="+std::to_string(begin)+
            "\nend_layer="+std::to_string(end)+
            "\nn_layers=10\nhc=2\nn_embd=6\nn_vocab=248320\nmax_cells=2048\nmax_t=8"
            "\ncpu_workers=4\nmax_cache_slots=16\nio_timeout_ms=30000"
            "\nfree_reserve_bytes=536870912\narena_budget_bytes=0\ngguf_shards="+gguf+"\n"+
            (ple ? "ple_gguf_path="+gguf+"\n" : "");
    }
    void write(const std::string& value) const {std::ofstream f(cfg); f<<value;}
};
}
int main() {
    try {
        Fixture f;
        HipWorkerConfig c;
        std::string error;
        f.write(f.config());
        need(read_hip_worker_config(f.cfg,c,error),"valid HIP v2 config");
        need(c.fields.stage_index==1 && c.fields.gguf_shards.size()==1,"parsed HIP identity");
        Plan p = parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10",10,{2,6},999);
        need(validate_hip_worker_config(c,p,1,error),"matches HIP plan, valid files");
        need(!validate_hip_worker_config(c,p,0,error),"CUDA index denied");
        need(!validate_hip_worker_config(c,p,2,error),"HIP index mismatch denied");
        auto bad=p; bad.model_fingerprint=42;
        need(!validate_hip_worker_config(c,bad,1,error),"model identity mismatch denied");
        bad=p;bad.stages[1].ordinal=1;
        need(!validate_hip_worker_config(c,bad,1,error),"HIP ordinal mismatch denied");
        bad=p;bad.geometry.hc=3;
        need(!validate_hip_worker_config(c,bad,1,error),"geometry mismatch denied");
        std::string v1="version=1\n"+f.config().substr(10);
        v1.erase(v1.find("gguf_shards=")); // valid CUDA v1 schema, not HIP-ready
        f.write(v1);
        HipWorkerConfig preserved=c;
        need(!read_hip_worker_config(f.cfg,preserved,error),"HIP rejects valid CUDA v1 config");
        need(error.find("v2")!=std::string::npos,"v2 requirement is explicit");
        need(preserved.fields.stage_index==c.fields.stage_index,"failed parse keeps output");
        f.write(f.config(1,3,7) + "gguf_shards=/tmp/dup.gguf\n");
        need(!read_hip_worker_config(f.cfg,preserved,error),"duplicated GGUF list rejected");
        f.write(f.config(1,1,7,true));
        need(read_hip_worker_config(f.cfg,preserved,error),"PLE stage config parses");
        auto ple_plan=parse_plan("cuda:0:0-1,hip:0:1-7,hip:1:7-10",10,{2,6},999);
        need(validate_hip_worker_config(preserved,ple_plan,1,error),"PLE stage preflight");
        f.write(f.config(1,1,7,false));
        need(!read_hip_worker_config(f.cfg,preserved,error),"missing PLE shard denied");
        f.write(f.config());
        {std::ofstream out(f.pack+"/index.txt",std::ios::app);out<<"corruption";}
        need(!validate_hip_worker_config(c,p,1,error),"pack-index tamper denied");
        need(error.find("SHA-256")!=std::string::npos,"digest mismatch diagnostic");
        std::cout << "HIP preflight config: PASS (" << assertions << " assertions; no HIP runtime)\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "HIP config: FAIL after " << assertions << ": " << ex.what() << '\n';
        return 1;
    }
}
