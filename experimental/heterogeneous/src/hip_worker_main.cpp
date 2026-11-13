// Exec-isolated HIP worker. NEVER link this executable into a CUDA process,
// the coordinator, or Strata's default engine executable.
#include "strata_heterogeneous/hip_worker_bootstrap.hpp"
#include "strata_heterogeneous/hip_worker_config.hpp"
#include "strata_heterogeneous/hip_stage_resources.hpp"
#include "strata_heterogeneous/hip_native_stage.hpp"
#include "strata_heterogeneous/window_admission.hpp"
#include "strata_heterogeneous/hip_file_experts.hpp"
#include "strata_heterogeneous/hip_verifier.hpp"
#include "strata_heterogeneous/hip_final_verifier.hpp"
#include "strata_heterogeneous/device.hpp"
#include "strata_heterogeneous/final_pipeline.hpp"
#include "strata/core/geometry.hpp"
#include "native_worker_defaults.hpp"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/socket.h>

#if !defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL) || defined(STRATA_ENABLE_SYCL)
#error "The M2.5 worker must link against an isolated wave32 HIP-only engine"
#endif

namespace hetero = strata::heterogeneous;
namespace {
bool check_socket(int fd) {
    int type=0; socklen_t len=sizeof(type);
    return ::getsockopt(fd,SOL_SOCKET,SO_TYPE,&type,&len)==0 && type==SOCK_STREAM;
}
// Do not advertise a prepared/ready worker on failed model or source setup.
// Best-effort failure frame uses v4 transport so the coordinator can report
// the failure immediately rather than waiting for its initialization deadline.
void startup_failure(int fd,const hetero::Plan& plan,size_t index,std::string what) noexcept {
    if(!check_socket(fd)||index>=plan.stages.size())return;
    try {
        hetero::Frame f;
        f.kind=hetero::Kind::failure;
        f.stage_index=static_cast<uint32_t>(index);
        f.request_id=1;f.window_id=1;f.rows=1;
        f.plan_fingerprint=plan.fingerprint();f.geometry=plan.geometry;
        f.error=what.empty()?"HIP worker initialization failed":std::move(what);
        if(f.error.size()>1024)f.error.resize(1024);
        hetero::send_frame(fd,f,1000);
    } catch(...) {} // parent may already have closed the socket
}
bool check_compiled_geometry(const hetero::HipWorkerConfig& config,const hetero::Plan& plan,
                             size_t index,std::string& error) {
    const auto& c=config.fields;
    const strata::core::ModelGeometry& g=strata::core::kCompiledGeometry;
    if(index>=plan.stages.size()||plan.stages[index].backend!=hetero::Backend::hip||
       g.n_layers!=c.n_layers||g.hc!=c.hc||g.n_embd!=c.n_embd||g.n_vocab!=c.n_vocab||
       g.n_expert!=512||g.n_expert_used!=10||g.qsa_interval!=4||
       g.ssm_state_size!=128||g.ssm_d_conv!=4) {
        error="HIP worker config does not match the compiled canonical Q2_0 geometry";
        return false;
    }
    // A single [0,n_layers) native reference stage now embeds tokens in
    // the final verifier (no earlier activation handoff is required).
    return true;
}
} // namespace

int main(int argc,char** argv) {
    try {
        if(argc==3&&std::string(argv[1])=="--check-config") {
            hetero::HipWorkerConfig c; std::string err;
            if(!hetero::read_hip_worker_config(argv[2],c,err)||
               !hetero::validate_cuda_worker_files(c.fields,err))throw std::runtime_error(err);
            std::cout<<"PASS: HIP worker config/index/profile preflight (no HIP runtime), stage "
                     <<c.fields.stage_index<<" ["<<c.fields.first_layer<<','<<c.fields.end_layer<<")\n";
            return 0;
        }
        hetero::HipWorkerInvocation invocation;
        std::string error;
        if(!hetero::parse_hip_worker_invocation(argc,const_cast<const char* const*>(argv),invocation,error))
            throw std::runtime_error(error);
        const auto& plan=invocation.plan;
        const size_t index=invocation.stage_index;
        if(!check_socket(3))throw std::runtime_error("HIP worker requires a supervisor-provided SOCK_STREAM on fd 3");
        // Errors after parsing the plan can be returned to its coordinator.
        try {
            hetero::HipWorkerConfig config;
            if(!hetero::read_hip_worker_config(invocation.config_path,config,error)||
               !hetero::validate_hip_worker_config(config,plan,index,error)||
               !check_compiled_geometry(config,plan,index,error))
                throw std::runtime_error(error);
            const auto& c=config.fields;
            // v1 is still readable for old CPU-only preflight tests, but the
            // production worker must not run its first window without the
            // original GGUF projections and layer-1 PLE table.
            if (c.version != 2)
                throw std::runtime_error("HIP worker needs v2 native GGUF/PLE configuration; v1 is preflight-only");
            // Explicit supervised worker launch is the opt-in for runtime use.
            // There is no second environment-variable approval handshake.
            strata_heterogeneous_native_worker_defaults();
            const auto& geometry=strata::core::kCompiledGeometry;
            hetero::HipStageResources owner(geometry,plan.model_fingerprint);
            hetero::HipStageLoadOptions lo;
            lo.pack_dir=c.pack_dir;
            lo.require_native_stage=true;
            lo.max_cells=c.max_cells;
            lo.experts_per_token=geometry.n_expert_used;
            lo.initial_position=0;  // fresh state ONLY; no prompt prefill
            lo.max_t=c.max_t;
            lo.timeout_ms=static_cast<int>(c.io_timeout_ms);
            lo.free_reserve_bytes=c.free_reserve_bytes;
            lo.arena_budget_bytes=c.arena_budget_bytes;
            if(!owner.load(plan,index,lo,error))throw std::runtime_error("weight/session startup: "+error);
            hetero::HipNativeStage native;
            hetero::HipNativeStageOptions no;
            no.gguf_shards=c.gguf_shards;
            no.ple_gguf_path=c.ple_gguf_path;
            no.free_reserve_bytes=c.free_reserve_bytes;
            no.total_arena_budget_bytes=c.arena_budget_bytes;
            if(!native.open(owner,plan,index,no,error))
                throw std::runtime_error("native dense/PLE startup: "+error);
            auto experts=std::make_unique<hetero::HipFileExperts>();
            hetero::HipFileExpertOptions eo;
            eo.pack_dir=c.pack_dir;
            eo.profile_path=c.profile_path;
            eo.max_cache_slots=c.max_cache_slots;
            eo.cpu_workers=static_cast<int>(c.cpu_workers);
            eo.free_reserve_bytes=c.free_reserve_bytes;
            if(!experts->open(plan,index,geometry,eo,error)||
               !owner.attach_experts(std::move(experts),error))
                throw std::runtime_error("expert provider startup: "+error);
            auto bindings=owner.make_bindings();
            const hetero::WindowLimits window_limits{
                0, static_cast<int64_t>(c.max_cells),
                static_cast<int64_t>(c.n_vocab), c.max_t};
            if(index+1==plan.stages.size()) {
                hetero::HipFinalVerifierExecutor final(bindings);
                hetero::serve_final_stage_worker(3,plan,index,final,
                    static_cast<int>(c.io_timeout_ms),invocation.shared_memory,
                    &window_limits);
                final.shutdown(); // before owner releases stage weights and experts
            } else {
                hetero::HipVerifierDriver driver(bindings);
                hetero::DeviceStageExecutor executor(driver);
                hetero::serve_worker_stateful(3,plan,index,executor,
                    static_cast<int>(c.io_timeout_ms),invocation.shared_memory,
                    &window_limits);
                executor.shutdown();driver.shutdown();
            }
            native.shutdown(); // after verifier graphs, before stage weights
            owner.shutdown();
            return 0;
        }catch(const std::exception& ex) {
            startup_failure(3,plan,index,ex.what());
            std::cerr<<"HIP worker: "<<ex.what()<<'\n';
            return 1;
        }
    } catch(const std::exception& ex) {
        std::cerr<<"HIP worker: "<<ex.what()<<'\n';
        return 2;
    }
}
