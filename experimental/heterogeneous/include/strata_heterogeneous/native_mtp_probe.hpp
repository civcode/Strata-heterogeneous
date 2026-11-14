#pragma once

// Explicit bounded HIP terminal qualification. One-shot mode logs proposals;
// relay mode returns bounded IDs to the target verifier. Vanilla MtpDrafter unchanged.
// Run the read-only asset audit first, against the exact configured target.
#include "strata_heterogeneous/hip_final_verifier.hpp"
#include "strata_heterogeneous/mtp_prefix_cursor.hpp"
#include "strata_heterogeneous/mtp_fresh_replay.hpp"
#include "strata_heterogeneous/mtp_three_round.hpp"
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

namespace strata::heterogeneous {
class NativeMtpProbe final : public HipFinalObserver {
public:
    NativeMtpProbe(std::string rt, std::vector<std::string> shards, uint64_t reserve,
                   uint64_t arena_budget, unsigned windows, unsigned relay_rounds = 0,
                   std::string trace_prefix = {}, bool reprime_tail = false, bool front_trace = false, bool repeat_round = false, bool fresh_replay = false, bool three_round = false, bool three_round_variable = false)
        : rt_(std::move(rt)), shards_(std::move(shards)), reserve_(reserve),
          arena_budget_(arena_budget), windows_(windows), relay_rounds_(relay_rounds),
          trace_prefix_(std::move(trace_prefix)), reprime_tail_(reprime_tail), front_trace_(front_trace), repeat_round_(repeat_round), fresh_replay_(fresh_replay), three_round_(three_round), three_round_variable_(three_round_variable) {}
    ~NativeMtpProbe() override { shutdown(); }

    bool prepare(const HipVerifierBindings& b, core::Verifier& terminal, std::string& error) override {
        const auto& g=b.geometry;
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
        if (relay_rounds_ > 1 && !three_round_) {
            error="MTP relay draft-state reuse failed fresh-prefix qualification";
            return false;
        }
        if (device_>=0 || !mtp_probe_window_count(windows_,relay_rounds_,fresh_replay_) ||
            (fresh_replay_ && (repeat_round_ || reprime_tail_)) || relay_rounds_>8 || arena_budget_ ||
            b.session.layer_hi!=g.n_layers || b.session.max_cells!=2048 ||
            g.n_layers!=48 || g.n_embd!=2560 || g.hc!=4 || g.n_expert!=512 ||
            g.n_vocab!=248320 || b.max_t<5 || b.max_t>8 ||
            core::native_embed() || std::getenv("STRATA_HEAD_HOST") ||
            !b.weights.find("token_embd.weight") || !b.weights.find("output.weight")) {
            error="MTP probe requires existing 2048-cell full-resident Q2_0 terminal geometry and no explicit arena cap or head spill";
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
            // even if mapped, exact runtime weights, plus 1 GiB for private
            // state/graphs/scratch. Never resize expert residency to fit.
            const uint64_t estimate=dense+experts+head_bytes*2+embed_bytes+(1ull<<30);
            size_t free_bytes=0, total_bytes=0;
            if (!ck(cudaMemGetInfo(&free_bytes,&total_bytes),error)) return false;
            std::fprintf(stderr,"HETERO_MTP_PROBE_ADMISSION device=%d free=%llu total=%llu reserve=%llu estimate=%llu head=%llu embed=%llu\n",
                         device_,static_cast<unsigned long long>(free_bytes),static_cast<unsigned long long>(total_bytes),
                         static_cast<unsigned long long>(reserve_),static_cast<unsigned long long>(estimate),
                         static_cast<unsigned long long>(head_bytes),static_cast<unsigned long long>(embed_bytes));
            if (!room(estimate,error)) return false;
            head_=std::make_unique<core::NativeHead>();
            if (!head_->load(shards_,g.n_embd,g.n_vocab,error) || head_->on_host() || !room(0,error)) {
                if(error.empty()) error="MTP probe native head spilled";
                return false;
            }
            embed_=std::make_unique<core::NativeEmbed>();
            if (!embed_->load(shards_,g.n_embd,g.n_vocab,error) || !room(0,error)) return false;
            core::set_native_embed(embed_.get());
            drafter_=std::make_unique<core::MtpDrafter>();
            drafter_->set_max_drafts(4);
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
            if (!trace_prefix_.empty()) trace_=std::make_unique<MtpStateTrace>(trace_prefix_,g);
            std::fprintf(stderr,"HETERO_MTP_PROBE_MODE state_trace=%d front_trace=%d repeat_round=%d relay_rounds=%u\n",
                         trace_?1:0,front_trace_?1:0,repeat_round_?1:0,relay_rounds_);
            std::fprintf(stderr,"HETERO_MTP_PROBE_LOADED device=%d draft_vram=%llu estimate=%llu\n",
                         device_,static_cast<unsigned long long>(drafter_->vram_bytes()),
                         static_cast<unsigned long long>(estimate));
            return true;
        } catch (const std::exception& ex) { error=ex.what(); return false; }
    }

    bool begin(const WindowSpec& spec,uint64_t request,uint64_t window,std::string& error) override {
        if (!cursor_ || request!=1 || window!=committed_+1 || window>windows_+relay_rounds_ ||
            (window>windows_ && !relay_pending_) ||
            (fresh_replay_ && spec.token_ids.size()!=1) || !ck(cudaSetDevice(device_),error)) {
            if(error.empty()) error="MTP probe window identity mismatch";
            return false;
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
        if (!cursor_ || request!=1 || window!=committed_+1 || window>windows_+relay_rounds_ ||
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
        if (!cursor_->committed(terminal.final_R_all(),keep,prime(),error)) return false;
        ++committed_;
        verified_next_=predictions[keep-1];
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
    bool relay_enabled() const noexcept override { return relay_rounds_!=0; }
    bool propose(const WindowSpec& anchor, WindowSpec& offer, std::string& error) override {
        if (!relay_rounds_ || relay_pending_ || committed_<windows_ || committed_>=windows_+relay_rounds_ ||
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
                if (n<1 || n>4 || result.generated!=n) {e="MTP relay invalid count";return false;}
                for(int i=0;i<n;++i)
                    if(tokens[i]<0 || tokens[i]>=n_vocab_ || !std::isfinite(probs[i]) ||
                       probs[i]<0.f || probs[i]>1.f) {e="MTP relay invalid candidate";return false;}
                candidates.assign(tokens.begin(),tokens.begin()+n);
                std::fprintf(stderr,"HETERO_MTP_RELAY_OFFER cell=%lld anchor=%d count=%d\n",
                             static_cast<long long>(cell),next,n);
                return true;
            },error);
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
    std::string trace_prefix_;
    bool reprime_tail_=false;
    bool front_trace_=false;
    bool repeat_round_=false;
    bool fresh_replay_=false;
    bool three_round_=false;
    bool three_round_variable_=false;
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
        return drafter_->draft(1,&next,cell,0,tokens,e,probs,0.f,&n,&result);
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
    std::function<bool(const float*,const int32_t*,size_t,int64_t,std::string&)> prime() {
        return [this](const float* rows,const int32_t* next,size_t n,int64_t cell,std::string& e) {
            if (!capture("prime_before",rows,next,n,cell,cell,e) || !room(512ull<<20,e) ||
                !drafter_->prefill(rows,next,static_cast<int64_t>(n),cell,e,true) ||
                !room(0,e) || !capture("prime_after",rows,next,n,cell,cell+int64_t(n),e)) return false;
            std::fprintf(stderr,"HETERO_MTP_PROBE_PRIMED cell=%lld rows=%zu\n",static_cast<long long>(cell),n);
            return true;
        };
    }
};
} // namespace strata::heterogeneous
