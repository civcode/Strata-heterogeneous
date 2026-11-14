// Exec-isolated HIP worker. NEVER link this executable into a CUDA process,
// the coordinator, or Strata's default engine executable.
#include "strata_heterogeneous/hip_worker_bootstrap.hpp"
#include "strata_heterogeneous/backend_domains.hpp"
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
#include "strata_heterogeneous/performance.hpp"

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
// Own every physical HIP stage separately, but in one native runtime.
// Destroy native dense companions before their weights/cache/session owners.
struct LoadedHipStage {
    hetero::HipStageResources owner;
    hetero::HipNativeStage native;
    explicit LoadedHipStage(uint64_t fingerprint)
        : owner(strata::core::kCompiledGeometry, fingerprint) {}
    void load(const hetero::Plan& plan, size_t index, const hetero::CudaWorkerConfig& c) {
        std::string error;
        const auto& geometry = strata::core::kCompiledGeometry;
        hetero::HipStageLoadOptions lo;
        lo.pack_dir = c.pack_dir;
        lo.require_native_stage = true;
        lo.max_cells = c.max_cells;
        lo.experts_per_token = geometry.n_expert_used;
        lo.max_t = c.max_t;
        lo.timeout_ms = static_cast<int>(c.io_timeout_ms);
        lo.free_reserve_bytes = c.free_reserve_bytes;
        lo.arena_budget_bytes = c.arena_budget_bytes;
        if (!owner.load(plan,index,lo,error)) throw std::runtime_error("weight/session startup: "+error);
        hetero::HipNativeStageOptions no;
        no.gguf_shards = c.gguf_shards;
        no.ple_gguf_path = c.ple_gguf_path;
        no.free_reserve_bytes = c.free_reserve_bytes;
        no.total_arena_budget_bytes = c.arena_budget_bytes;
        if (!native.open(owner,plan,index,no,error)) throw std::runtime_error("native dense/PLE startup: "+error);
        auto experts = std::make_unique<hetero::HipFileExperts>();
        hetero::HipFileExpertOptions eo;
        eo.pack_dir = c.pack_dir;
        eo.profile_path = c.profile_path;
        eo.max_cache_slots = c.max_cache_slots;
        eo.cpu_workers = static_cast<int>(c.cpu_workers);
        eo.free_reserve_bytes = c.free_reserve_bytes;
        if (!experts->open(plan,index,geometry,eo,error) || !owner.attach_experts(std::move(experts),error))
            throw std::runtime_error("expert provider startup: "+error);
    }
};
void serve_native_hip_domain(const hetero::HipWorkerInvocation& invocation, bool compact, int startup_wait_ms) {
    const auto& plan = invocation.plan;
    const auto domain = hetero::worker_domains(plan,true).back();
    if (invocation.stage_index != domain.terminal_stage() || invocation.domain_configs.size() != domain.size())
        throw std::runtime_error("HIP domain invocation mismatch");
    std::vector<hetero::HipWorkerConfig> configs;
    std::string error;
    for (size_t j = 0; j < domain.size(); ++j) {
        hetero::HipWorkerConfig config;
        const size_t i = domain.first_stage + j;
        if (!hetero::read_hip_worker_config(invocation.domain_configs[j],config,error) ||
            !hetero::validate_hip_worker_config(config,plan,i,error) ||
            !check_compiled_geometry(config,plan,i,error))
            throw std::runtime_error("HIP domain config: "+error);
        const auto& c = config.fields;
        const uint64_t slots = uint64_t(c.end_layer-c.first_layer)*strata::core::kCompiledGeometry.n_expert;
        if (c.max_cache_slots != slots)
            throw std::runtime_error("initial native HIP domain requires explicit full residency");
        if (!configs.empty()) {
            const auto& first = configs.front().fields;
            if (c.pack_dir != first.pack_dir || c.gguf_shards != first.gguf_shards ||
                c.profile_path != first.profile_path || c.index_sha256 != first.index_sha256 ||
                c.max_cells != first.max_cells || c.max_t != first.max_t ||
                c.io_timeout_ms != first.io_timeout_ms)
                throw std::runtime_error("HIP domain requires identical model, context and window limits");
        }
        configs.push_back(std::move(config));
    }
    std::vector<std::unique_ptr<LoadedHipStage>> owners;
    std::vector<hetero::HipNativeChainStage> bindings;
    for (size_t j = 0; j < domain.size(); ++j) {
        auto owner = std::make_unique<LoadedHipStage>(plan.model_fingerprint);
        owner->load(plan,domain.first_stage+j,configs[j].fields);
        bindings.push_back({owner->owner.make_bindings(),configs[j].fields.free_reserve_bytes});
        owners.push_back(std::move(owner));
    }
    const auto& c = configs.front().fields;
    const hetero::WindowLimits limits{0,static_cast<int64_t>(c.max_cells),static_cast<int64_t>(c.n_vocab),c.max_t};
    hetero::HipFinalVerifierExecutor final(std::move(bindings),{!compact});
    // Same native final-result handler; one external commit ACK covers the entire HIP chain.
    hetero::serve_final_stage_worker(3,plan,invocation.stage_index,final,static_cast<int>(c.io_timeout_ms),
        invocation.shared_memory,&limits,startup_wait_ms,true);
    final.shutdown();
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
            if (!invocation.domain_configs.empty()) {
                const int startup_wait_ms = hetero::worker_startup_wait_ms();
                const bool compact = hetero::performance_flag("STRATA_HETERO_COMPACT_RESULT");
                (void)hetero::performance_flag("STRATA_HETERO_FAST_FENCE");
                (void)hetero::performance_flag("STRATA_HETERO_PERF_TRACE");
                if (compact && !strata::core::Verifier::nan_guard_on())
                    throw std::runtime_error("compact result requires native GPU nonfinite guard");
                strata_heterogeneous_native_worker_defaults();
                serve_native_hip_domain(invocation,compact,startup_wait_ms);
                return 0;
            }
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
            const int startup_wait_ms = hetero::worker_startup_wait_ms();
            const bool compact = hetero::performance_flag("STRATA_HETERO_COMPACT_RESULT");
            (void)hetero::performance_flag("STRATA_HETERO_FAST_FENCE");
            (void)hetero::performance_flag("STRATA_HETERO_PERF_TRACE");
            if(compact && index+1==plan.stages.size() && !strata::core::Verifier::nan_guard_on())
                throw std::runtime_error("compact result requires native GPU nonfinite guard");
            strata_heterogeneous_native_worker_defaults();
            LoadedHipStage loaded(plan.model_fingerprint);
            loaded.load(plan,index,c);
            auto bindings=loaded.owner.make_bindings();
            const hetero::WindowLimits window_limits{
                0, static_cast<int64_t>(c.max_cells),
                static_cast<int64_t>(c.n_vocab), c.max_t};
            if(index+1==plan.stages.size()) {
                hetero::HipFinalVerifierExecutor final(bindings, {!compact});
                hetero::serve_final_stage_worker(3,plan,index,final,
                    static_cast<int>(c.io_timeout_ms),invocation.shared_memory,
                    &window_limits, startup_wait_ms);
                final.shutdown(); // before owner releases stage weights and experts
            } else {
                hetero::HipVerifierDriver driver(bindings);
                hetero::DeviceStageExecutor executor(driver);
                hetero::serve_worker_stateful(3,plan,index,executor,
                    static_cast<int>(c.io_timeout_ms),invocation.shared_memory,
                    &window_limits, startup_wait_ms);
                executor.shutdown();driver.shutdown();
            }
            loaded.native.shutdown(); // after verifier graphs, before stage weights
            loaded.owner.shutdown();
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
