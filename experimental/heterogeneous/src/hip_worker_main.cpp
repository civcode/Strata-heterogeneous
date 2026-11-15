// Exec-isolated HIP worker. NEVER link this executable into a CUDA process,
// the coordinator, or Strata's default engine executable.
#include "strata_heterogeneous/hip_worker_bootstrap.hpp"
#include "strata_heterogeneous/backend_domains.hpp"
#include "strata_heterogeneous/hip_worker_config.hpp"
#include "strata_heterogeneous/hip_stage_resources.hpp"
#include "strata_heterogeneous/hip_native_stage.hpp"
#include "strata_heterogeneous/window_admission.hpp"
#include "strata_heterogeneous/direct_verified_registration.hpp"
#include "strata_heterogeneous/hip_file_experts.hpp"
#include "strata_heterogeneous/hip_verifier.hpp"
#include "strata_heterogeneous/hip_final_verifier.hpp"
#include "strata_heterogeneous/device.hpp"
#include "strata_heterogeneous/final_pipeline.hpp"
#include "strata/core/geometry.hpp"
#include "native_worker_defaults.hpp"
#include "strata_heterogeneous/performance.hpp"
#include "strata_heterogeneous/native_prefill_domain.hpp"
#include "strata_heterogeneous/native_mtp_probe.hpp"

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
    hetero::NativePrefillExpertView pf_resources;
    explicit LoadedHipStage(uint64_t fingerprint)
        : owner(strata::core::kCompiledGeometry, fingerprint) {}
    void load(const hetero::Plan& plan, size_t index, const hetero::CudaWorkerConfig& c, bool extended = false) {
        std::string error;
        const auto& geometry = strata::core::kCompiledGeometry;
        auto capacity = [&](const char* phase) {
            if (extended && !hetero::sample_native_capacity(static_cast<int>(c.device_ordinal),
                    index,phase,c.free_reserve_bytes,0,error)) throw std::runtime_error(error);
        };
        capacity("before_load");
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
        std::cerr << "HETERO_ARENA stage=" << index << " backend=HIP session_bytes=" << owner.session_bytes()
                  << " weight_bytes=" << owner.weight_bytes() << " reserve_bytes=" << c.free_reserve_bytes << '\n';
        capacity("weight_session_loaded");
        hetero::HipNativeStageOptions no;
        no.gguf_shards = c.gguf_shards;
        no.ple_gguf_path = c.ple_gguf_path;
        no.free_reserve_bytes = c.free_reserve_bytes;
        no.total_arena_budget_bytes = c.arena_budget_bytes;
        if (!native.open(owner,plan,index,no,error)) throw std::runtime_error("native dense/PLE startup: "+error);
        capacity("native_dense_ple_loaded");
        auto experts = std::make_unique<hetero::HipFileExperts>();
        hetero::HipFileExpertOptions eo;
        eo.pack_dir = c.pack_dir;
        eo.profile_path = c.profile_path;
        eo.max_cache_slots = c.max_cache_slots;
        eo.cpu_workers = static_cast<int>(c.cpu_workers);
        eo.free_reserve_bytes = c.free_reserve_bytes;
        if (!experts->open(plan,index,geometry,eo,error))
            throw std::runtime_error("expert provider startup: "+error);
        pf_resources=experts->native_prefill_resources();
        if (!owner.attach_experts(std::move(experts),error))
            throw std::runtime_error("expert provider startup: "+error);
        capacity("experts_loaded");
    }
};
void serve_native_hip_domain(const hetero::HipWorkerInvocation& invocation, bool compact, int startup_wait_ms) {
    const auto& plan = invocation.plan;
    const auto domain = hetero::worker_domains(plan,true).back();
    if (invocation.stage_index != domain.terminal_stage() || invocation.domain_configs.size() != domain.size())
        throw std::runtime_error("HIP domain invocation mismatch");
    const bool extended = invocation.long_context_chunk != 0;
    const bool mtp_service = hetero::performance_flag("STRATA_HETERO_MTP_SERVICE");
    const bool mtp_mixed = hetero::direct_verified_plan(plan) && plan.stages.front().end_layer==15;
    const bool mtp_hip_pair = plan.total_layers==48 && plan.stages.size()==2 &&
        plan.stages[0].backend==hetero::Backend::hip && plan.stages[0].ordinal==0 &&
        plan.stages[0].first_layer==0 && plan.stages[0].end_layer==24 &&
        plan.stages[1].backend==hetero::Backend::hip && plan.stages[1].ordinal==1 &&
        plan.stages[1].first_layer==24 && plan.stages[1].end_layer==48;
    if (mtp_service && (invocation.long_context_chunk>hetero::MtpRequestPolicy::bulk_chunk_limit ||
        invocation.kv!=hetero::KvMode::int8 || !invocation.shared_memory ||
        (!mtp_mixed && !mtp_hip_pair)))
        throw std::runtime_error("MTP service candidate requires shared Q8 mixed 15/18/15 or HIP-only 24/24");
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
        if ((extended && !hetero::long_context_stage_valid(c.max_cells,c.free_reserve_bytes,
                c.arena_budget_bytes,c.max_t,c.max_cache_slots,c.end_layer-c.first_layer)) ||
            (!extended && c.max_cells > hetero::legacy_workload_context))
            throw std::runtime_error("invalid HIP long-context capacity or missing worker capability");
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
        if (mtp_service && (!hetero::MtpRequestPolicy::capacity_valid(c.max_cells) ||
            extended!=hetero::MtpRequestPolicy::requires_long_context(c.max_cells) || c.max_t!=8 || c.free_reserve_bytes < (1ull<<30) ||
            c.arena_budget_bytes || c.index_sha256!=hetero::mtp_qualification_index))
            throw std::runtime_error("MTP service requires approved aligned capacity, matching long-context mode, full residency and reserve");
        configs.push_back(std::move(config));
    }
    std::vector<std::unique_ptr<LoadedHipStage>> owners;
    std::vector<hetero::HipNativeChainStage> bindings;
    std::vector<hetero::NativePrefillStage> pf_stages;
    for (size_t j = 0; j < domain.size(); ++j) {
        auto owner = std::make_unique<LoadedHipStage>(plan.model_fingerprint);
        owner->load(plan,domain.first_stage+j,configs[j].fields,extended || mtp_service);
        auto b=owner->owner.make_bindings();
        strata_heterogeneous_verify_kv(b.session,invocation.kv,domain.first_stage+j);
        if (extended || mtp_service) {
            strata_heterogeneous_verify_full_experts(b.geometry,b.session,b.hits);
            const auto stage = domain.first_stage+j;
            const int device = static_cast<int>(configs[j].fields.device_ordinal);
            const auto reserve = configs[j].fields.free_reserve_bytes;
            const auto chunk = invocation.long_context_chunk;
            b.capacity_probe = [stage,device,reserve,chunk,&g=b.geometry,&s=b.session](const char* phase,std::string& e) {
                if (chunk && std::string_view(phase) == "verifier_ready")
                    return hetero::check_native_prefill_capacity(device,stage,phase,g,s,chunk,reserve,e);
                return hetero::sample_native_capacity(device,stage,phase,reserve,0,e);
            };
        }
        bindings.push_back({b,configs[j].fields.free_reserve_bytes});
        pf_stages.push_back({b.weights,b.geometry,b.session,owner->pf_resources,
            configs[j].fields.free_reserve_bytes,configs[j].fields.arena_budget_bytes,invocation.long_context_chunk});
        owners.push_back(std::move(owner));
    }
    const auto& c = configs.front().fields;
    const hetero::WindowLimits limits{0,static_cast<int64_t>(c.max_cells),static_cast<int64_t>(c.n_vocab),c.max_t};
    const bool overlap = hetero::performance_flag("STRATA_HETERO_HIP_SPLIT_OVERLAP");
    const bool arm = hetero::performance_flag("STRATA_HETERO_RECEIVER_ARM");
    const bool direct_verified = hetero::performance_flag("STRATA_HETERO_DIRECT_VERIFIED");
    const bool guarded_direct = hetero::performance_flag("STRATA_HETERO_GUARDED_DIRECT");
    if (guarded_direct && (!direct_verified || c.max_cells!=2048 || c.free_reserve_bytes<1073741824ull ||
        configs.back().fields.free_reserve_bytes<1073741824ull))
        throw std::runtime_error("guarded HIP requires direct verified 2K capacity and 1 GiB reserve per owner");
    if (direct_verified && (!invocation.shared_memory || !overlap || arm || compact || c.max_t != 8 ||
        !hetero::direct_verified_plan(plan) || std::getenv("STRATA_HETERO_MTP_PROBE_RT")))
        throw std::runtime_error("direct verified HIP requires mixed shared full-logit native overlap with max_t=8");
    if (!hetero::MtpRequestPolicy::long_worker_mode_valid(extended,mtp_service,overlap,arm,
        direct_verified,std::getenv("STRATA_HETERO_MTP_PROBE_RT") != nullptr))
        throw std::runtime_error("long-context rejects diagnostic overlap, receiver arm, direct path and non-service MTP");
    if (arm && (!overlap || compact)) throw std::runtime_error("receiver arm requires native HIP overlap and full logits");
    std::unique_ptr<hetero::NativeMtpProbe> mtp;
    if (const char* path = std::getenv("STRATA_HETERO_MTP_PROBE_RT"); path && *path) {
        if (compact || !overlap || arm) throw std::runtime_error("one-shot MTP probe requires HIP overlap without external arm");
        const char* count = std::getenv("STRATA_HETERO_MTP_PROBE_WINDOWS");
        const auto n = count ? std::string(count) : std::string();
        const unsigned windows=hetero::mtp_probe_parse_windows(n);
        unsigned relay=0;
        if (const char* raw=std::getenv("STRATA_HETERO_MTP_RELAY_ROUNDS")) {
            const std::string value(raw);
            if(value.size()!=1 || value[0]<'0' || value[0]>'8') throw std::runtime_error("invalid MTP relay round count");
            relay=unsigned(value[0]-'0');
        }
        const bool fresh_replay=hetero::performance_flag("STRATA_HETERO_MTP_FRESH_REPLAY");
        const bool three_round=hetero::performance_flag("STRATA_HETERO_MTP_THREE_ROUND");
        const bool three_round_variable=hetero::performance_flag("STRATA_HETERO_MTP_THREE_ROUND_VARIABLE");
        if (three_round_variable && !three_round) throw std::runtime_error("variable prefix requires three-round mode");
        if (!hetero::mtp_probe_window_count(windows,relay,fresh_replay))
            throw std::runtime_error("invalid MTP probe window count or fresh replay mode");
        const char* raw_prefix=std::getenv("STRATA_HETERO_MTP_STATE_PREFIX");
        const std::string trace_prefix=raw_prefix?raw_prefix:"";
        const bool reprime=hetero::performance_flag("STRATA_HETERO_MTP_REPRIME_TAIL");
        const bool front_trace=hetero::performance_flag("STRATA_HETERO_MTP_FRONT_TRACE");
        const bool repeat_round=hetero::performance_flag("STRATA_HETERO_MTP_REPEAT_NATIVE_ROUND");
        if (repeat_round && (!front_trace || relay)) throw std::runtime_error("invalid MTP repeat mode");
        if (three_round && (!hetero::mtp_three_shape(windows,relay,fresh_replay,repeat_round,reprime,three_round_variable) ||
            (trace_prefix.empty()==front_trace))) throw std::runtime_error("invalid three-round diagnostic mode");
        if ((!trace_prefix.empty() && (trace_prefix.front()!='/' || trace_prefix.size()>4096 || (relay>1 && !three_round))) ||
            (reprime && (trace_prefix.empty() || relay!=1)) || (front_trace && trace_prefix.empty())) throw std::runtime_error("invalid bounded MTP state trace");
        const char* raw_budget=std::getenv("STRATA_HETERO_MTP_GENERATE_MAX_NEW");
        const char* raw_stops=std::getenv("STRATA_HETERO_MTP_GENERATE_STOPS");
        std::vector<int32_t> service_stops;
        if (mtp_service) service_stops=hetero::mtp_generation_policy(1,"1",raw_stops?raw_stops:"").stops;
        auto generation=hetero::mtp_generation_policy(windows,raw_budget?raw_budget:"0",mtp_service?"":raw_stops?raw_stops:"");
        const bool resident_generation=hetero::performance_flag("STRATA_HETERO_MTP_RESIDENT_GENERATION");
        if (resident_generation && !generation.enabled() && !mtp_service) throw std::runtime_error("resident MTP requires explicit generation policy");
        const auto& last=configs.back().fields;
        if (generation.enabled() || mtp_service) {
            if (relay || fresh_replay || three_round || three_round_variable || reprime || repeat_round ||
                (trace_prefix.empty()==front_trace))
                throw std::runtime_error("invalid MTP generation qualification mode");
            if (!hetero::validate_mtp_qualification_runtime(path,last.index_sha256,error))
                throw std::runtime_error(error);
        }
        mtp=std::make_unique<hetero::NativeMtpProbe>(path,last.gguf_shards,last.free_reserve_bytes,
                                                  last.arena_budget_bytes,windows,relay,trace_prefix,reprime,front_trace,repeat_round,fresh_replay,three_round,three_round_variable,std::move(generation),resident_generation,std::move(service_stops));
    }
    if (mtp_service && !mtp) throw std::runtime_error("MTP service runtime missing");
    hetero::DirectHandoffRegistration registration;
    hetero::HipFinalVerifierExecutor final(std::move(bindings),{!compact,overlap,arm,guarded_direct});
    hetero::DirectSharedBinder bind_slot;
    if (direct_verified) bind_slot = [&](const hetero::SharedActivation& slot, std::string& e) {
        auto b = hetero::register_verified_slot(registration,slot,static_cast<int>(plan.stages[domain.first_stage].ordinal),plan);
        b.guarded=guarded_direct;
        return final.bind_direct_input(b,e);
    };
    if (mtp) final.set_observer(mtp.get());
    std::cerr << "HETERO_RECEIVER_ARM enabled=" << arm << '\n';
    std::cerr << "HETERO_HIP_SPLIT_OVERLAP enabled=" << overlap << '\n';
    // Same native final-result handler; one external commit ACK covers the entire HIP chain.
    hetero::NativePrefillHandler prefill;
    if (!direct_verified) prefill = [&](const hetero::Frame& cmd,int fd,std::string& e) {
            if (mtp_service) {
                if (!mtp || !mtp->begin_bulk_prompt(cmd,e)) return false;
                return hetero::run_native_prefill_domain(cmd,fd,pf_stages,false,
                    static_cast<int>(c.io_timeout_ms),e,{},{},{},
                    [&](const float* r,int64_t rows,int64_t pos,std::string& err) {
                        return mtp->seed_bulk_prompt(r,rows,pos,err);
                    }) && final.adopt_native_prefill(cmd.request_id,cmd.rows,e);
            }
            return hetero::run_native_prefill_domain(cmd,fd,pf_stages,false,
                       static_cast<int>(c.io_timeout_ms),e) &&
                   final.adopt_native_prefill(cmd.request_id,cmd.rows,e);
        };
    try {
        hetero::serve_final_stage_worker(3,plan,invocation.stage_index,final,static_cast<int>(c.io_timeout_ms),
            invocation.shared_memory,&limits,startup_wait_ms,true,std::move(prefill),std::move(bind_slot),guarded_direct);
    } catch (...) {
        final.shutdown();
        if (!registration.release_after_drain()) std::_Exit(70);
        throw;
    }
    final.shutdown();
    if (!registration.release_after_drain()) throw std::runtime_error("direct HIP unregister failed");
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
                strata_heterogeneous_native_worker_defaults(invocation.kv);
                serve_native_hip_domain(invocation,compact,startup_wait_ms);
                return 0;
            }
            if (hetero::performance_flag("STRATA_HETERO_DIRECT_VERIFIED") ||
                hetero::performance_flag("STRATA_HETERO_HIP_SPLIT_OVERLAP") ||
                hetero::performance_flag("STRATA_HETERO_RECEIVER_ARM") ||
                std::getenv("STRATA_HETERO_MTP_PROBE_RT"))
                throw std::runtime_error("HIP split overlap requires the grouped worker");
            hetero::HipWorkerConfig config;
            if(!hetero::read_hip_worker_config(invocation.config_path,config,error)||
               !hetero::validate_hip_worker_config(config,plan,index,error)||
               !check_compiled_geometry(config,plan,index,error))
                throw std::runtime_error(error);
            const auto& c=config.fields;
            if (c.max_cells > hetero::legacy_workload_context)
                throw std::runtime_error("large context requires grouped long-context worker");
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
            strata_heterogeneous_native_worker_defaults(invocation.kv);
            LoadedHipStage loaded(plan.model_fingerprint);
            loaded.load(plan,index,c);
            auto bindings=loaded.owner.make_bindings();
            strata_heterogeneous_verify_kv(bindings.session,invocation.kv,index);
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
