#pragma once

// Explicit bounded HIP terminal qualification. One-shot mode logs proposals;
// relay mode returns bounded IDs to the target verifier. Vanilla MtpDrafter unchanged.
// Run the read-only asset audit first, against the exact configured target.
#include "strata_heterogeneous/hip_final_verifier.hpp"
#include "strata_heterogeneous/mtp_prefix_cursor.hpp"
#include "strata_heterogeneous/mtp_bulk_seed_stats.hpp"
#include "strata_heterogeneous/performance.hpp"
#include "strata_heterogeneous/mtp_fresh_replay.hpp"
#include "strata_heterogeneous/mtp_three_round.hpp"
#include "strata_heterogeneous/mtp_generation.hpp"
#include "strata_heterogeneous/mtp_resident.hpp"
#include "strata_heterogeneous/mtp_request_policy.hpp"
#include "strata_heterogeneous/speculation.hpp"
#include "strata_heterogeneous/mtp_state_trace.hpp"
#include "strata/artifact/gguf_reader.hpp"
// Vanilla PinnedRing intentionally discards cleanup/error-clear return values;
// HIP annotates them nodiscard and this worker builds with -Werror.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "strata/core/mtp.hpp"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include "strata/core/native_head.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <utility>
#include <cstring>

namespace strata::heterogeneous {
class NativeMtpProbe final : public HipFinalObserver {
public:
    NativeMtpProbe(std::string rt, std::vector<std::string> shards, uint64_t reserve,
                   uint64_t arena_budget, unsigned windows, unsigned relay_rounds = 0,
                   std::string trace_prefix = {}, bool reprime_tail = false, bool front_trace = false, bool repeat_round = false, bool fresh_replay = false, bool three_round = false, bool three_round_variable = false, MtpGenerationPolicy generation = {}, bool resident_generation = false, std::vector<int32_t> service_stops = {})
        : rt_(std::move(rt)), shards_(std::move(shards)), reserve_(reserve),
          arena_budget_(arena_budget), windows_(windows), relay_rounds_(relay_rounds),
          trace_prefix_(std::move(trace_prefix)), reprime_tail_(reprime_tail), front_trace_(front_trace), repeat_round_(repeat_round), fresh_replay_(fresh_replay), three_round_(three_round), three_round_variable_(three_round_variable), generation_(std::move(generation)), resident_generation_(resident_generation), request_(resident_generation?0:1), service_stops_(std::move(service_stops)) {}
    ~NativeMtpProbe() override { shutdown(); }

    bool prepare(const HipVerifierBindings& b, core::Verifier& terminal, std::string& error) override {
        const auto& g=b.geometry;
        native_window_=performance_flag("STRATA_HETERO_MTP_NATIVE_WINDOW");
        if (service_enabled()) {
            const char* width=std::getenv("STRATA_HETERO_MTP_SERVICE_WINDOW");
            if (width && std::strcmp(width,"0")!=0) {
                if (std::strcmp(width,"4")==0) service_candidates_=3;
                else if (std::strcmp(width,"5")==0) service_candidates_=4;
                else {error="invalid native MTP window";return false;}
                native_window_=true; // ordinary native confidence window, not a profiling experiment
            }
        }
        if (native_window_ && !service_enabled()) {
            error="native confidence window requires resident MTP service";return false;
        }
        // Fresh-prefix qualification found divergent subsequent native offers
        // after rejection. Do not reuse speculative draft state until fixed.
        if (!trace_prefix_.empty() && (std::getenv("STRATA_PARK_ASYNC") || std::getenv("STRATA_PARK_VRAM_MIB"))) {
            error="MTP state trace needs ordinary synchronous native snapshot configuration"; return false;
        }
        if (front_trace_ && trace_prefix_.empty()) {error="MTP front trace needs state trace";return false;}
        if (repeat_round_ && (!front_trace_ || relay_rounds_)) {error="MTP repeat needs one-shot front trace";return false;}
        // Sole exception: explicit nonresident bounded qualification.
        if (three_round_variable_ && !three_round_) {
            error="variable prefix requires three-round mode"; return false;
        }
        if (three_round_ && (!mtp_three_shape(windows_,relay_rounds_,fresh_replay_,repeat_round_,reprime_tail_,three_round_variable_) ||
            (trace_prefix_.empty()==front_trace_))) {
            error="invalid MTP three-round diagnostic"; return false;
        }
        if (service_enabled() && (!resident_generation_ || generation_.enabled() || relay_rounds_ ||
            three_round_ || three_round_variable_ || fresh_replay_ || repeat_round_ || reprime_tail_ || front_trace_ ||
            !trace_prefix_.empty() || reserve_ < (1ull<<30) || !MtpRequestPolicy{static_cast<uint32_t>(b.session.max_cells),1,4,248320,{0},service_stops_}.valid())) {
            error="invalid experimental resident MTP service configuration";return false;
        }
        if (resident_generation_ && !generation_.enabled() && !service_enabled()) {error="resident MTP requires bounded generation";return false;}
        if (generation_.enabled() && (!generation_.valid() || generation_.prompt_rows!=windows_ ||
            relay_rounds_ || three_round_ || three_round_variable_ || fresh_replay_ || repeat_round_ || reprime_tail_ ||
            (trace_prefix_.empty()==front_trace_))) {
            error="invalid MTP generation qualification"; return false;
        }
        if (relay_rounds_ > 1 && !three_round_) {
            error="MTP relay draft-state reuse failed fresh-prefix qualification";
            return false;
        }
        if (device_>=0 || !mtp_probe_window_count(windows_,relay_rounds_,fresh_replay_) ||
            (fresh_replay_ && (repeat_round_ || reprime_tail_)) || relay_rounds_>8 || arena_budget_ ||
            b.session.layer_hi!=g.n_layers ||
            (service_enabled() ? !MtpRequestPolicy::capacity_valid(b.session.max_cells) : b.session.max_cells!=2048) ||
            g.n_layers!=48 || g.n_embd!=2560 || g.hc!=4 || g.n_expert!=512 ||
            g.n_vocab!=248320 || b.max_t<5 || b.max_t>8 ||
            core::native_embed() || std::getenv("STRATA_HEAD_HOST") ||
            !b.weights.find("token_embd.weight") || !b.weights.find("output.weight")) {
            error="MTP probe requires admitted full-resident Q2_0 terminal geometry/capacity and no explicit arena cap or head spill";
            return false;
        }
        try {
            const std::filesystem::path dir(rt_);
            const auto dense=std::filesystem::file_size(dir/"dense.bin");
            const auto experts=std::filesystem::file_size(dir/"experts.bin");
            const auto vocab=std::filesystem::file_size(dir/"draft_vocab.bin");
            if (!std::filesystem::is_regular_file(dir/"dense.txt") || !dense || dense>(512ull<<20) ||
                experts!=uint64_t(g.n_expert)*strata::kernels::cpu::BLOB ||
                !vocab || vocab%4 || vocab>uint64_t(g.n_vocab)*4) {
                error="MTP probe runtime file extent mismatch"; return false;
            }
            const strata::GgufModel model(shards_);
            size_t at=0;
            const auto* output=model.find("output.weight",&at);
            const auto* embedding=model.find("token_embd.weight",&at);
            if (!output || !embedding) { error="MTP probe target tensors missing"; return false; }
            const uint64_t head_bytes=strata::tensor_payload_bytes(*output);
            const uint64_t embed_bytes=strata::tensor_payload_bytes(*embedding);
            if (!head_bytes || !embed_bytes || head_bytes>(3ull<<30) || embed_bytes>(3ull<<30)) {
                error="MTP probe unexpected target head/embedding extent"; return false;
            }
            device_=terminal.device();
            if (!ck(cudaSetDevice(device_),error)) return false;
            // Conservative bounded admission estimate, not a general native
            // allocator estimator: count head twice (subset/repack), embed
            // even if mapped, exact runtime weights, plus 1 GiB for fixed
            // scratch/graphs. Account for context-dependent allocations
            // explicitly for service and retain the old cushion. Never resize
            // expert residency to fit. This is not a transient peak model.
            uint64_t context_bytes=0;
            if (service_enabled()) {
                context_bytes=core::MtpDrafter::resident_context_bytes(g,b.session.max_cells,b.max_t);
                if (!context_bytes || b.session.qsa_states[b.session.qsa_primary()].max_cells!=b.session.max_cells) {
                    error="MTP service requires sized fully resident Q8 draft context";return false;
                }
            }
            const uint64_t estimate=dense+experts+head_bytes*2+embed_bytes+(1ull<<30)+context_bytes;
            size_t free_bytes=0, total_bytes=0;
            if (!ck(cudaMemGetInfo(&free_bytes,&total_bytes),error)) return false;
            std::fprintf(stderr,"HETERO_MTP_PROBE_ADMISSION device=%d free=%llu total=%llu reserve=%llu estimate=%llu head=%llu embed=%llu context_bytes=%llu\n",
                         device_,static_cast<unsigned long long>(free_bytes),static_cast<unsigned long long>(total_bytes),
                         static_cast<unsigned long long>(reserve_),static_cast<unsigned long long>(estimate),
                         static_cast<unsigned long long>(head_bytes),static_cast<unsigned long long>(embed_bytes),
                         static_cast<unsigned long long>(context_bytes));
            if (!room(estimate,error)) return false;
            head_=std::make_unique<core::NativeHead>();
            if (!head_->load(shards_,g.n_embd,g.n_vocab,error) || head_->on_host() || !room(0,error)) {
                if(error.empty()) error="MTP probe native head spilled";
                return false;
            }
            embed_=std::make_unique<core::NativeEmbed>();
            if (!embed_->load(shards_,g.n_embd,g.n_vocab,error) || !room(0,error)) return false;
            // In a full HIP domain the layer-zero verifier is on HIP0 and
            // may lazily capture graphs after this observer is initialized.
            // Publishing HIP1's embed globally would change its target math
            // and could hand it a device address owned by another card.
            // Historical diagnostics keep their original global binding;
            // resident service uses the same native table privately.
            drafter_=std::make_unique<core::MtpDrafter>();
            if (service_enabled()) drafter_->set_embedding_override(embed_.get());
            else core::set_native_embed(embed_.get());
            drafter_->set_max_drafts(service_enabled() ? service_candidates_ : 4);
            // Separate state on the terminal device. No cross-runtime device
            // address and no mutation of the verified target's head.
            if (!drafter_->load(rt_,g,b.session,b.max_t,error,b.session.max_cells) || !room(0,error)) return false;
            stride_=size_t(g.hc)*size_t(g.n_embd);
            const size_t bytes=stride_*size_t(b.max_t)*sizeof(float);
            if (!room(drafter_->bind_bytes(head_->row_bytes(),g.n_vocab)+bytes+(512ull<<20),error) ||
                !ck(cudaMalloc(reinterpret_cast<void**>(&scratch_),bytes),error) ||
                !drafter_->bind(b.weights,head_.get(),scratch_,error) ||
                !ck(cudaDeviceSynchronize(),error) || !room(0,error)) return false;
            if (front_trace_ && (!room(core::mtp_front_bytes,error) ||
                !ck(cudaMalloc(&front_device_,core::mtp_front_bytes),error) ||
                !drafter_->set_front_trace(front_device_,core::mtp_front_bytes,error) || !room(0,error))) return false;
            cursor_=std::make_unique<MtpPrefixCursor>(stride_,b.session.max_cells,g.n_vocab,b.max_t);
            n_vocab_=g.n_vocab;
            geometry_=&g;
            context_=b.session.max_cells;
            max_t_=b.max_t;
            if (service_enabled() && (core::native_embed()!=nullptr || !drafter_->experts_resident())) {
                error="MTP service requires isolated draft embedding and resident draft experts";return false;
            }
            if (resident_generation_ && !resident_kv_supported()) {error="resident MTP draft KV format/residency mismatch";return false;}
            if (!resident_generation_ && !trace_prefix_.empty()) trace_=std::make_unique<MtpStateTrace>(trace_prefix_,g);
            std::fprintf(stderr,"HETERO_MTP_PROBE_MODE state_trace=%d front_trace=%d repeat_round=%d relay_rounds=%u\n",
                         !trace_prefix_.empty()?1:0,front_trace_?1:0,repeat_round_?1:0,relay_rounds_);
            if (generation_.enabled()) std::fprintf(stderr,"HETERO_MTP_GENERATION_MODE prompt=%u max_new=%u\n",
                         generation_.prompt_rows,generation_.max_new);
            std::fprintf(stderr,"HETERO_MTP_PROBE_LOADED device=%d draft_vram=%llu estimate=%llu\n",
                         device_,static_cast<unsigned long long>(drafter_->vram_bytes()),
                         static_cast<unsigned long long>(estimate));
            return true;
        } catch (const std::exception& ex) { error=ex.what(); return false; }
    }

    bool resident_relay_enabled() const noexcept override {return resident_generation_;}
    bool service_enabled() const noexcept override {return !service_stops_.empty();}
    bool configure_request(const Frame& meta,std::string& error) override {
        if (!service_enabled() || !cursor_ || !drafter_ || !request_ || meta.request_id!=request_ ||
            meta.kind!=Kind::mtp_service_request || !meta.window || meta.window_id || meta.window->pos0 ||
            committed_ || cursor_->committed_position() || service_policy_ || !resident_kv_supported()) {
            error="MTP service request identity or state refused";return false;
        }
        MtpRequestPolicy policy{static_cast<uint32_t>(context_),meta.mtp_max_new,service_candidates_,static_cast<uint32_t>(n_vocab_),meta.window->token_ids,service_stops_};
        if (meta.rows!=policy.prompt.size()) {error="invalid MTP service prompt or budget";return false;}
        auto admitted=admit_mtp_service_request(std::move(policy));
        if (!admitted) {error="invalid MTP service prompt or budget";return false;}
        service_policy_=std::move(admitted);
        drafter_->set_prompt_len(static_cast<int64_t>(service_policy_->prompt.size()));
        return true;
    }
    // Called on the worker control thread before any target bulk execution.
    bool begin_bulk_prompt(const Frame& cmd,std::string& error) {
        if (!service_enabled() || !cursor_ || !drafter_ || !request_ || cmd.request_id!=request_ ||
            cmd.kind!=Kind::prefill_spec || cmd.window_id!=1 || !cmd.window || cmd.window->pos0!=0 ||
            cmd.rows!=cmd.window->token_ids.size() || committed_ || bulk_active_ ||
            cursor_->committed_position() || relay_pending_ || verified_next_!=-1 ||
            !resident_kv_supported() || !service_policy_ || !service_policy_->bulk_prefix(cmd.window->token_ids,cmd.prefill_chunk)) {
            error="MTP bulk prompt metadata or state refused";return false;
        }
        if (!cursor_->start_bulk_prompt(service_policy_->prompt,error)) return false;
        bulk_stats_={};
        const char* detail=std::getenv("STRATA_HETERO_MTP_BULK_TRACE");
        bulk_detail_=cmd.rows<=128 || (detail && std::strcmp(detail,"1")==0);
        bulk_active_=true;
        bulk_chunk_=cmd.prefill_chunk;
        return true;
    }
    // Native Prefill::on_chunk runs on the terminal owner's thread with its
    // producer stream already fenced. Synchronous prime consumes R before
    // the native prefill scratch is reused; no residual pointer is retained.
    bool seed_bulk_prompt(prefill::Prefill& pf,const float* rows,int64_t count,int64_t pos,std::string& error) {
        int current=-1;
        if (!bulk_active_ || !cursor_ || !drafter_ || !service_policy_ || !rows || !bulk_chunk_ ||
            pos!=cursor_->committed_position() || pos<0 || pos>=int64_t(service_policy_->prompt.size())-1 ||
            count!=std::min<int64_t>(bulk_chunk_,int64_t(service_policy_->prompt.size())-1-pos) ||
            !ck(cudaGetDevice(&current),error) || current!=device_) {
            if(error.empty()) error="MTP bulk residual owner/state refused";
            return false;
        }
        // Explicitly opt in until native execution and acceptance have been
        // characterized. Keep the proven per-group path as the default.
        static const bool batch_enabled=[] {
            const char* value=std::getenv("STRATA_HETERO_MTP_BATCH");
            return value && std::strcmp(value,"1")==0;
        }();
        return bulk_stats_.seed(*cursor_,pos,rows,size_t(count),prime(bulk_detail_),
            [&](const float* r,const int32_t* next,size_t n,int64_t cell,std::string& e) {
                if (!batch_enabled || n<64 || trace_ || front_trace_)
                    return MtpBulkSeedResult::declined;
                if (!embed_ || !resident_kv_supported() || !room(MtpRequestPolicy::prime_headroom_bytes,e)) {
                    if(e.empty()) e="local MTP batch lacks admitted embedding or resident cache";
                    return MtpBulkSeedResult::failed;
                }
                const char* why=nullptr;
                if (!pf.draft_kv(*drafter_,r,next,int64_t(n),cell,e,&why,embed_.get())) {
                    if (!e.empty()) return MtpBulkSeedResult::failed;
                    // Keep the first applicability reason even when quiet.
                    if (bulk_detail_ || bulk_stats_.narrow_rows==0)
                        std::fprintf(stderr,"HETERO_MTP_BULK_BATCH_DECLINED cell=%lld rows=%zu reason=%s\n",
                                     static_cast<long long>(cell),n,why?why:"not applicable");
                    return MtpBulkSeedResult::declined;
                }
                if (!room(0,e)) return MtpBulkSeedResult::failed;
                if (bulk_detail_)
                    std::fprintf(stderr,"HETERO_MTP_BULK_BATCH_PRIMED cell=%lld rows=%zu\n",
                                 static_cast<long long>(cell),n);
                return MtpBulkSeedResult::completed;
            },error);
    }
    bool finish_bulk_prompt(uint64_t request,uint32_t rows,std::string& error) override {
        if (!bulk_active_ || !cursor_ || !drafter_ || request!=request_ || committed_ ||
            !service_policy_ || rows+1!=service_policy_->prompt.size() ||
            cursor_->committed_position()!=rows || !ck(cudaSetDevice(device_),error) ||
            !drafter_->idle(error) || !room(0,error)) {
            if(error.empty()) error="MTP bulk prompt drain/adoption refused";
            return false;
        }
        const auto& s=bulk_stats_;
        std::fprintf(stderr,"HETERO_MTP_BULK_SEED_SUMMARY request=%llu rows=%llu chunks=%llu "
                     "batch_calls=%llu batch_rows=%llu narrow_calls=%llu narrow_rows=%llu declines=%llu "
                     "callback_ns=%llu batch_ns=%llu narrow_ns=%llu decline_ns=%llu\n",
                     static_cast<unsigned long long>(request),
                     static_cast<unsigned long long>(s.rows()),static_cast<unsigned long long>(s.chunks),
                     static_cast<unsigned long long>(s.batch_calls),static_cast<unsigned long long>(s.batch_rows),
                     static_cast<unsigned long long>(s.narrow_calls),static_cast<unsigned long long>(s.narrow_rows),
                     static_cast<unsigned long long>(s.declines),static_cast<unsigned long long>(s.callback_ns),
                     static_cast<unsigned long long>(s.batch_ns),static_cast<unsigned long long>(s.narrow_ns),
                     static_cast<unsigned long long>(s.decline_ns));
        bulk_active_=false;
        bulk_chunk_=0;
        committed_=1; // one native prefill protocol window, not one per chunk
        return true;
    }
    bool reset_request(uint64_t request,std::string& error) override {
        if (!resident_generation_ || (service_enabled() ? (!request || request_==UINT64_MAX || request!=request_+1) :
            !mtp_resident_next_request(request_,request)) || !geometry_ ||
            !drafter_ || !cursor_ || !resident_kv_supported() || drafter_->chain_live() ||
            !ck(cudaSetDevice(device_),error) || !drafter_->idle(error)) {
            if(error.empty()) error="resident MTP reset identity/state/drain refused";
            return false;
        }
        // Existing native sequence zero, in place on the owning stream.
        // No graph/weight reallocation, no change to inference operations.
        // Target sessions are separately reset and fenced by the executor.
        drafter_->set_source_R(nullptr);
        drafter_->set_prompt_len(0);
        drafter_->set_cell_floor(0);
        core::qsa_state_zero(drafter_->kv_state(),*geometry_,drafter_->stream());
        if (!ck(cudaGetLastError(),error) || !drafter_->idle(error) || !room(0,error)) return false;
        // A terminal unsubmitted offer or a cancelled committed tail has
        // no successor in this request. Drop the borrowed pointer and IDs.
        cursor_=std::make_unique<MtpPrefixCursor>(stride_,context_,n_vocab_,max_t_);
        trace_.reset();
        if (!trace_prefix_.empty())
            trace_=std::make_unique<MtpStateTrace>(trace_prefix_+".r"+std::to_string(request),*geometry_);
        committed_=0;
        bulk_active_=false;
        bulk_chunk_=0;
        bulk_stats_={};
        bulk_detail_=false;
        service_policy_.reset();
        verified_next_=-1;
        relay_pending_=false;
        request_=request;
        std::fprintf(stderr,"HETERO_MTP_RESIDENT_RESET request=%llu\n",static_cast<unsigned long long>(request));
        return true;
    }

    bool begin(const WindowSpec& spec,uint64_t request,uint64_t window,std::string& error) override {
        if (!cursor_ || !request_ || request!=request_ || window!=committed_+1 || window>window_limit() ||
            (!service_enabled() && window>windows_ && !relay_pending_) ||
            (fresh_replay_ && spec.token_ids.size()!=1) || !ck(cudaSetDevice(device_),error)) {
            if(error.empty()) error="MTP probe window identity mismatch";
            return false;
        }
        if (service_enabled()) {
            if (!service_policy_ || (spec.pos0 < int64_t(service_policy_->prompt.size()) ?
                !service_policy_->prompt_window(spec.pos0,spec.token_ids) :
                (!service_policy_->decode_window(spec.pos0,verified_next_,spec.token_ids) ||
                 (!relay_pending_ && (spec.token_ids.size()!=1 || service_policy_->can_offer(spec.pos0,verified_next_)))))) {
                error="MTP service window outside admitted prompt or decode policy";return false;
            }
        }
        if (generation_.enabled()) {
            if (window<=windows_ ? (spec.pos0!=int64_t(window-1) || spec.token_ids.size()!=1)
                                 : !generation_.live_window(spec.pos0,spec.token_ids)) {
                error="MTP generation window outside admitted prefix/stop/length"; return false;
            }
        }
        if (three_round_) {
            const bool prompt=window<=windows_;
            if ((prompt && (spec.pos0!=int64_t(window-1) || spec.token_ids.size()!=1 ||
                            (!three_round_variable_ && spec.token_ids.front()!=mtp_three_prompt[window-1]))) ||
                (!prompt && !mtp_three_window_fits(spec.pos0,spec.token_ids.size(),unsigned(window-windows_-1),windows_))) {
                error="MTP three-round window bound or prompt mismatch"; return false;
            }
        }
        if (!cursor_->begin(spec.pos0,spec.token_ids,prime(),error)) return false;
        relay_pending_=false;
        return true;
    }
    bool committed(const WindowSpec& spec,uint64_t request,uint64_t window,uint32_t keep,
                   core::Verifier& terminal,const std::vector<int32_t>& predictions,std::string& error) override {
        if (!cursor_ || !request_ || request!=request_ || window!=committed_+1 || window>window_limit() ||
            terminal.device()!=device_ || predictions.size()!=spec.token_ids.size() ||
            !keep || keep>predictions.size() || !ck(cudaSetDevice(device_),error)) {
            if(error.empty()) error="MTP probe commit identity mismatch";
            return false;
        }
        // Coordinator enforces the same budget BEFORE sending any commit.
        // This is a defensive observer check, not a distributed rollback.
        if (three_round_ && window>windows_ &&
            !mtp_three_commit_fits(spec.pos0,keep,unsigned(window-windows_-1),windows_)) {
            error="MTP three-round committed prefix exceeds diagnostic budget"; return false;
        }
        if (generation_.enabled() && (window<=windows_ ? keep!=1 :
             (!generation_.live_window(spec.pos0,spec.token_ids) ||
              keep!=greedy_accepted_prefix(spec.token_ids,predictions)))) {
            error="invalid MTP generation commit"; return false;
        }
        if (service_enabled() && (!service_policy_ || (spec.pos0 < int64_t(service_policy_->prompt.size()) ?
            keep!=spec.token_ids.size() : keep!=greedy_accepted_prefix(spec.token_ids,predictions)))) {
            error="MTP service commit differs from admitted prefix";return false;
        }
        if (!cursor_->committed(terminal.final_R_all(),keep,prime(),error)) return false;
        ++committed_;
        verified_next_=predictions[keep-1];
        if (generation_.enabled() || service_enabled()) return true; // service has no terminal diagnostic draft
        // Finish the sole live relay with a second native draft for diagnosis
        // ONLY, even without tracing. Never submit another candidate window.
        // No front taps or state readbacks are added to the trace-free path.
        if (committed_!=(relay_rounds_ ? windows_+relay_rounds_ : windows_)) return true;
        return cursor_->finish(predictions[keep-1],
            [this](const float* row,int32_t next,int64_t cell,std::string& e) {
                // In repeat mode the exact same native pair is evaluated
                // twice, without a target launch, prime, or advance between.
                // No physical rollback or reset. Neither offer is submitted.
                for(unsigned attempt=0;attempt<(repeat_round_?2u:1u);++attempt) {
                    std::array<int32_t,8> tokens{};
                    std::array<float,8> probs{};
                    int n=0;
                    core::MtpDrafter::DraftResult result;
                    if (!capture("draft_before",row,&next,1,cell,cell,e) ||
                        !room(512ull<<20,e) ||
                        !draft_ordered(row,next,cell,tokens.data(),probs.data(),n,result,e) ||
                        !room(0,e) || !capture("draft_after",row,&next,1,cell,cell+1,e)) return false;
                    if (n<1 || n>4 || result.generated!=n) {e="MTP probe invalid generated count";return false;}
                    for(int i=0;i<n;++i)
                        if(tokens[i]<0 || tokens[i]>=n_vocab_ || !std::isfinite(probs[i]) ||
                           probs[i]<0.f || probs[i]>1.f) {e="MTP probe invalid candidate";return false;}
                    std::fprintf(stderr,"HETERO_MTP_PROBE_DRAFT cell=%lld verified_next=%d count=%d tokens=",
                                 static_cast<long long>(cell),next,n);
                    for(int i=0;i<n;++i) std::fprintf(stderr,"%d,",tokens[i]);
                    std::fprintf(stderr," (not submitted to target)\n");
                    if(repeat_round_) std::fprintf(stderr,"HETERO_MTP_REPEAT_NATIVE_ROUND attempt=%u\n",attempt+1);
                }
                return true;
            },error);
    }
    bool relay_enabled() const noexcept override { return relay_rounds_!=0 || generation_.enabled() || service_enabled(); }
    bool anchor_only_enabled() const noexcept override { return generation_.enabled() || service_enabled(); }
    bool propose(const WindowSpec& anchor, WindowSpec& offer, std::string& error) override {
        if (!relay_enabled() || !request_ || relay_pending_ || committed_<windows_ ||
            (service_enabled() && (!service_policy_ || !service_policy_->can_offer(anchor.pos0,verified_next_))) ||
            (!service_enabled() && !generation_.enabled() && committed_>=windows_+relay_rounds_) ||
            (generation_.enabled() && !generation_.can_draft(anchor.pos0)) ||
            !cursor_ || anchor.pos0!=cursor_->committed_position() ||
            anchor.token_ids!=std::vector<int32_t>{verified_next_} ||
            !ck(cudaSetDevice(device_),error)) {
            if(error.empty()) error="MTP relay not admitted";
            return false;
        }
        offer.pos0=anchor.pos0;
        const bool ok=cursor_->offer(verified_next_,offer.token_ids,
            [this](const float* row,int32_t next,int64_t cell,std::vector<int32_t>& candidates,std::string& e) {
                std::array<int32_t,8> tokens{};
                std::array<float,8> probs{};
                int n=0;
                core::MtpDrafter::DraftResult result;
                if (!capture("draft_before",row,&next,1,cell,cell,e) ||
                    !room(512ull<<20,e) ||
                    !draft_ordered(row,next,cell,tokens.data(),probs.data(),n,result,e) ||
                    !room(0,e) || !capture("draft_after",row,&next,1,cell,cell+1,e)) return false;
                // Experimental diagnostic only: rewrite the exact accepted
                // pair with vanilla prefill, keeping the already-made offer.
                if (reprime_tail_ && (!drafter_->prefill(row,&next,1,cell,e,true) ||
                    !room(0,e) || !capture("reprime_after",row,&next,1,cell,cell+1,e))) return false;
                if (n<1 || n>(service_enabled()?static_cast<int>(service_candidates_):4) || result.generated!=n) {e="MTP relay invalid count";return false;}
                for(int i=0;i<n;++i)
                    if(tokens[i]<0 || tokens[i]>=n_vocab_ || !std::isfinite(probs[i]) ||
                       probs[i]<0.f || probs[i]>1.f) {e="MTP relay invalid candidate";return false;}
                // Native draft() reports generated rows, including the row
                // that ended a confidence-gated chain. Native window sizing
                // decides which rows to actually submit; the wire adapter
                // then clips only for request context/stop/length admission.
                // Never add a separate acceptance-history policy.
                const int submitted=native_window_ ?
                    core::pipeline_window_size(probs.data(),0,n,
                        static_cast<int>(service_candidates_)+1,service_min_p_) - 1 : n;
                candidates.assign(tokens.begin(),tokens.begin()+submitted);
                if (service_enabled()) {
                    auto clipped=service_policy_->clip_offer(cell+1,next,candidates);
                    candidates.assign(clipped.begin()+1,clipped.end());
                }
                if (generation_.enabled()) {
                    // Record the original native list for independent fresh replay.
                    // Truncate only the outgoing live IDs, not the native draft.
                    std::string line="HETERO_MTP_GENERATION_OFFER cell="+std::to_string(cell)+
                        " anchor="+std::to_string(next)+" tokens=";
                    for (auto id:candidates) line+=std::to_string(id)+",";
                    const auto count=generation_.candidate_count(cell+1,next,candidates);
                    line+=" submitted="+std::to_string(count);
                    std::fprintf(stderr,"%s\n",line.c_str());
                    candidates.resize(count);
                }
                std::fprintf(stderr,"HETERO_MTP_RELAY_OFFER cell=%lld anchor=%d count=%d\n",
                             static_cast<long long>(cell),next,n);
                if (native_window_ && performance_trace())
                    std::fprintf(stderr,"HETERO_MTP_NATIVE_WINDOW request=%llu cell=%lld generated=%d confidence_admitted=%d submitted=%zu\n",
                                 static_cast<unsigned long long>(request_),static_cast<long long>(cell),
                                 n,submitted,candidates.size());
                return true;
            },error,generation_.enabled() || service_enabled());
        relay_pending_=ok;
        return ok;
    }
    void shutdown() noexcept override {
        if(device_<0) return;
        // Every native call above is synchronous. No explicit frees on an
        // unconfirmed runtime drain. Draft graphs die before their inputs.
        if(cudaSetDevice(device_)!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess) std::_Exit(70);
        drafter_.reset();
        if(front_device_ && cudaFree(front_device_)!=cudaSuccess) std::_Exit(70);
        front_device_=nullptr;
        cursor_.reset();
        if(scratch_ && cudaFree(scratch_)!=cudaSuccess) std::_Exit(70);
        scratch_=nullptr;
        if(core::native_embed()==embed_.get()) core::set_native_embed(nullptr);
        embed_.reset();
        head_.reset();
        device_=-1;
    }
private:
    std::string rt_;
    std::vector<std::string> shards_;
    uint64_t reserve_,arena_budget_;
    unsigned windows_,relay_rounds_=0,committed_=0;
    int32_t verified_next_=-1;
    bool relay_pending_=false;
    bool native_window_=false;
    uint32_t service_candidates_=4;
    static constexpr float service_min_p_=0.5f;
    std::string trace_prefix_;
    bool reprime_tail_=false;
    bool front_trace_=false;
    bool repeat_round_=false;
    bool fresh_replay_=false;
    bool three_round_=false;
    bool three_round_variable_=false;
    MtpGenerationPolicy generation_;
    bool resident_generation_=false;
    uint64_t request_=1;
    std::vector<int32_t> service_stops_;
    std::unique_ptr<const MtpRequestPolicy> service_policy_;
    bool bulk_active_=false;
    uint32_t bulk_chunk_=0;
    MtpBulkSeedStats bulk_stats_;
    bool bulk_detail_=false;
    const core::ModelGeometry* geometry_=nullptr;
    int64_t context_=0;
    size_t max_t_=0;
    unsigned window_limit() const noexcept { return service_enabled() ? (service_policy_?static_cast<unsigned>(service_policy_->feed_end()):0u) : generation_.enabled()?generation_.feed_end():windows_+relay_rounds_; }
    void* front_device_=nullptr;
    std::unique_ptr<MtpStateTrace> trace_;
    int device_=-1;
    int64_t n_vocab_=0;
    size_t stride_=0;
    float* scratch_=nullptr;
    std::unique_ptr<core::NativeHead> head_;
    std::unique_ptr<core::NativeEmbed> embed_;
    std::unique_ptr<core::MtpDrafter> drafter_;
    std::unique_ptr<MtpPrefixCursor> cursor_;
    bool resident_kv_supported() const noexcept {
        if (!drafter_) return false;
        const auto& state=drafter_->kv_state();
        if (service_enabled()) {
            // Native MtpDrafter inherits the requested target INT8 setting.
            // Verify the actual independent draft state, not only the flag.
            return state.kv_mode==0 && state.kv_elastic==-1 && state.kv_int8 && !state.kv_q4 &&
                   !state.kv_hybrid && !state.kv_rot && state.k_q && state.v_q && state.k_scale && state.v_scale &&
                   !state.k_pool && !state.v_pool && state.page_table && state.n_pages>0 &&
                   state.n_slots==state.n_pages && !state.host.present() && state.max_cells==context_;
        }
        return state.kv_mode==0 && state.kv_elastic<0 && !state.kv_int8 && !state.kv_q4 &&
               !state.kv_hybrid && state.k_pool && state.v_pool && state.n_slots==state.n_pages;
    }
    // Use vanilla's stream-ordered source staging. draft_first stages through
    // a default-stream device copy, which is not ordered with the drafter's
    // non-default stream on the observed HIP runtime. The verifier row stays
    // live until the synchronous draft returns; clear the borrowed pointer on
    // both success and failure, including exceptions.
    bool draft_ordered(const float* row,int32_t next,int64_t cell,int32_t* tokens,float* probs,
                       int& n,core::MtpDrafter::DraftResult& result,std::string& e) {
        struct ClearSource {
            core::MtpDrafter& d;
            ~ClearSource() { d.set_source_R(nullptr); }
        } clear{*drafter_};
        drafter_->set_source_R(row);
        return drafter_->draft(1,&next,cell,0,tokens,e,probs,service_enabled()?service_min_p_:0.f,&n,&result);
    }
    bool ck(cudaError_t rc,std::string& e) const {
        if(rc==cudaSuccess) return true;
        e=std::string("MTP probe HIP: ")+cudaGetErrorString(rc); return false;
    }
    bool room(uint64_t bytes,std::string& e) const {
        size_t free=0,total=0;
        if(!ck(cudaMemGetInfo(&free,&total),e)) return false;
        if(reserve_>free || bytes>uint64_t(free)-reserve_) {
            e="MTP probe exceeds remaining VRAM above existing reserve: free="+std::to_string(free)+
              " reserve="+std::to_string(reserve_)+" requested="+std::to_string(bytes);
            return false;
        }
        return true;
    }
    bool capture(const char* phase,const float* rows,const int32_t* next,size_t n,
                 int64_t cell,int64_t upto,std::string& e) {
        return !trace_ || trace_->capture(phase,drafter_->kv_state(),rows,next,n,cell,upto,e,
            std::strcmp(phase,"draft_after")==0?front_device_:nullptr);
    }
    std::function<bool(const float*,const int32_t*,size_t,int64_t,std::string&)> prime(bool detail=true) {
        return [this,detail](const float* rows,const int32_t* next,size_t n,int64_t cell,std::string& e) {
            if (!capture("prime_before",rows,next,n,cell,cell,e) || !room(MtpRequestPolicy::prime_headroom_bytes,e) ||
                !drafter_->prefill(rows,next,static_cast<int64_t>(n),cell,e,true) ||
                !room(0,e) || !capture("prime_after",rows,next,n,cell,cell+int64_t(n),e)) return false;
            if (detail)
                std::fprintf(stderr,"HETERO_MTP_PROBE_PRIMED cell=%lld rows=%zu\n",static_cast<long long>(cell),n);
            return true;
        };
    }
};
} // namespace strata::heterogeneous
