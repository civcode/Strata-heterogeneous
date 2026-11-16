// CPU-only final-worker protocol. Real forked worker processes, no GPU model.
#include "strata_heterogeneous/final_pipeline.hpp"
#include "strata_heterogeneous/backend_domains.hpp"
#include "strata_heterogeneous/speculation.hpp"
#include "strata_heterogeneous/shared.hpp"
#include "strata_heterogeneous/window_admission.hpp"
#include <algorithm>
#include <cerrno>
#include <iostream>
#include <cmath>
#include <cstring>
#include <limits>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
using namespace strata::heterogeneous;
namespace {
int assertions = 0;
void check(bool condition, const char* why) {
    ++assertions;
    if (!condition) throw std::runtime_error(why);
}
template <typename F> void fails(F&& fn, const char* why) {
    bool threw = false;
    try { fn(); } catch (const std::exception&) { threw = true; }
    check(threw, why);
}
Frame input_frame(const Plan& plan, uint64_t request = 17, uint64_t window = 1) {
    Frame frame;
    frame.request_id = request; frame.window_id = window;
    frame.rows = 3; frame.geometry = plan.geometry;
    frame.plan_fingerprint = plan.fingerprint();
    frame.payload.assign(3 * plan.geometry.floats_per_row(), 0.25f);
    return frame;
}
class MockFinal final : public FinalStageExecutor {
public:
    bool bad_identity = false;
    bool grouped = false;
    int arm_mode = 0;
    bool armed = false;
    bool receiver_arm_enabled() const noexcept override { return arm_mode != 0; }
    bool arm_window(std::string& e) override {
        if (arm_mode != 1 || armed || tokens_.empty()) { e = "synthetic arm refusal"; return false; }
        armed = true; return true;
    }
    int mtp_mode = 0;
    int service_fault = 0; // 1 metadata, 2 proposal, 3 post-target commit observer
    bool mtp_relay_enabled() const noexcept override { return mtp_mode != 0; }
    bool mtp_anchor_only_enabled() const noexcept override { return mtp_mode == 5 || mtp_mode == 7 || mtp_mode == 8; }
    bool mtp_resident_enabled() const noexcept override { return mtp_mode == 7 || mtp_mode == 8; }
    bool mtp_service_enabled() const noexcept override { return mtp_mode == 8; }
    bool configure_mtp_request(const Frame& meta,std::string& e) override {
        if (service_fault==1) {e="CPU service metadata refusal";return false;}
        if (mtp_mode!=8 || !meta.window || !meta.mtp_max_new) {e="mock service metadata refused";return false;}
        return true;
    }
    bool propose_mtp(const WindowSpec& anchor, WindowSpec& offer, std::string& e) override {
        if (mtp_mode==2 || service_fault==2) {e="CPU proposal refusal";return false;}
        offer=anchor;
        if (mtp_mode==5 || mtp_mode==6) return true;
        offer.token_ids.push_back((anchor.token_ids.front()+17)%1000);
        offer.token_ids.push_back(mtp_mode==3 ? 1000 : 0);
        if(mtp_mode==4) ++offer.pos0;
        return true;
    }
    bool nonfinite = false;
    bool fail_commit = false;
    bool fail_reset = false;
    float unmarked_logit = 0.f;
    bool prepare(const Plan& plan, size_t i, std::string& e) override {
        index_ = static_cast<uint32_t>(i);
        plan_ = plan;
        return mock_.prepare(plan, i, e);
    }
    bool reset_request(uint64_t request, std::string& e) override {
        if (fail_reset) { e = "CPU protocol reset refused"; return false; }
        tokens_.clear();
        return mock_.reset_request(request, e);
    }
    bool begin_window(const WindowSpec& spec, uint64_t r, uint64_t w, std::string& e) override {
        tokens_ = spec.token_ids;
        armed = false;
        return mock_.begin_window(spec, r, w, e);
    }
    bool run_staged_window(const Frame&, ActivationView, std::string& e) override {
        e = "final worker must use run_final_window";
        return false;
    }
    bool run_window(const Frame&, Frame&, std::string& e) override {
        e = "final worker must not create an activation reply";
        return false;
    }
    bool run_final_window(const Frame& m, ActivationView input,
                          FinalStageResult& result, std::string& e) override {
        if (arm_mode && !armed) { e = "mock receiver was not armed"; return false; }
        // Verify that real intermediate protocol workers handed off every layer,
        // not just that token metadata reached the final mock.
        float expected = 0.25f;
        for (uint32_t layer = 0; layer < plan_.stages[grouped ? index_-1 : index_].first_layer; ++layer)
            expected += static_cast<float>(layer + 1) * 0.0001f;
        ByteRange all{input.data(), input.bytes()};
        for (size_t i = 0; i < input.bytes() / sizeof(float); ++i) {
            if (all.read_f32(i) != expected) {
                e = "incorrect intermediate activation at final stage";
                return false;
            }
        }
        // Host-only mock: use the ordinary stage's deterministic byte transform,
        // then fabricate token predictions strictly for transport testing.
        if (!mock_.run_staged_window(m, input, e)) return false;
        result.stage_index = index_;
        result.request_id = m.request_id + (bad_identity ? 1 : 0);
        result.window_id = m.window_id;
        result.plan_fingerprint = plan_.fingerprint();
        result.rows = m.rows;
        result.vocab_size = 1000;
        for (int32_t token : tokens_) result.predicted_tokens.push_back((token + 17) % 1000);
        if (nonfinite) result.nonfinite_rows.push_back(1);
        if (!std::isfinite(unmarked_logit)) {
            result.logits.emplace(static_cast<size_t>(result.rows) * result.vocab_size, 0.f);
            result.logits->back() = unmarked_logit;
        }
        if (!tokens_.empty() && tokens_[0] == 9) {
            result.logits.emplace(static_cast<size_t>(result.rows) * result.vocab_size, 0.f);
            (*result.logits)[0] = -0.f;
            (*result.logits)[1] = 1.25f;
        }
        return true;
    }
    bool commit_window(uint64_t, uint64_t, std::string& e) override {
        e = "legacy full commit forbidden"; return false;
    }
    bool commit_prefix(uint64_t r, uint64_t w, uint32_t k, std::string& e) override {
        if (fail_commit) { e = "injected final commit failure"; return false; }
        if (!mock_.commit_prefix(r,w,k,e)) return false;
        if (service_fault==3) {e="CPU observer refused after target commit";return false;}
        return true;
    }
    void abandon_window() noexcept override { mock_.abandon_window(); }
    void shutdown() noexcept override { mock_.shutdown(); }
private:
    MockStageExecutor mock_;
    Plan plan_;
    uint32_t index_ = 0;
    std::vector<int32_t> tokens_;
};
// CPU-only negative fixture: corrupt the producer output after execution,
// before the real shared-activation provisional gate examines it.
class NonfiniteProducer final : public StageExecutor {
    MockStageExecutor inner_;
public:
    bool prepare(const Plan& p, size_t i, std::string& e) override { return inner_.prepare(p,i,e); }
    bool reset_request(uint64_t r, std::string& e) override { return inner_.reset_request(r,e); }
    bool begin_window(const WindowSpec& s, uint64_t r, uint64_t w, std::string& e) override { return inner_.begin_window(s,r,w,e); }
    bool run_window(const Frame& i, Frame& o, std::string& e) override { return inner_.run_window(i,o,e); }
    bool run_staged_window(const Frame& m, ActivationView v, std::string& e) override {
        if (!inner_.run_staged_window(m,v,e)) return false;
        const float bad = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(v.data(), &bad, sizeof(bad));
        return true;
    }
    bool commit_window(uint64_t r, uint64_t w, std::string& e) override { return inner_.commit_window(r,w,e); }
    bool commit_prefix(uint64_t r, uint64_t w, uint32_t k, std::string& e) override { return inner_.commit_prefix(r,w,k,e); }
    void abandon_window() noexcept override { inner_.abandon_window(); }
    void shutdown() noexcept override { inner_.shutdown(); }
};
struct WorkerGroup {
    WorkerGroup() = default;
    WorkerGroup(const WorkerGroup&) = delete;
    WorkerGroup& operator=(const WorkerGroup&) = delete;
    WorkerGroup(WorkerGroup&&) noexcept = default;
    WorkerGroup& operator=(WorkerGroup&&) noexcept = default;
    std::vector<int> sockets;
    std::vector<pid_t> pids;
    ~WorkerGroup() {
        for (int fd : sockets) ::close(fd);
        for (pid_t pid : pids) {
            int status = 0;
            if (::waitpid(pid, &status, WNOHANG) == 0) {
                // A broken protocol must never hang the test runner.
                ::kill(pid, SIGKILL);
                (void)::waitpid(pid, &status, 0);
            }
        }
    }
};
WorkerGroup launch(const Plan& plan, bool shared, bool bad = false,
                   bool nonfinite = false, bool fail_commit = false,
                   bool wrong_mode = false, float unmarked_logit = 0.f,
                   bool resident = false, bool fail_reset = false,
                   bool delayed_startup = false, bool grouped = false, bool wrong_group_hello = false,
                   bool bad_prefill = false, int arm_mode = 0, int mtp_mode = 0, bool isolated_prefill = false,
                   bool direct_verified = false, bool bad_producer = false, bool guarded_direct = false,
                   int64_t max_cells = 64, int service_fault = 0) {
    WorkerGroup group;
    for (const auto& domain : worker_domains(plan,grouped)) {
        const size_t i = domain.terminal_stage();
        int fd[2]{};
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) != 0) throw std::runtime_error("socketpair");
        const pid_t pid = ::fork();
        if (pid < 0) throw std::runtime_error("fork");
        if (pid == 0) {
            for (int previous : group.sockets) ::close(previous);
            ::close(fd[0]);
            try {
                const WindowLimits limits{0, max_cells, 1000, 8};
                DirectSharedBinder bind;
                if (direct_verified) bind=[](const SharedActivation& slot,std::string& error) {
                    if (!slot.valid() || slot.descriptor()<0) {
                        error="invalid mock direct descriptor"; return false;
                    }
                    return true;
                };
                NativePrefillHandler pf;
                if (grouped && !direct_verified) pf=[&,i](const Frame& cmd,int sock,std::string& error) {
                    if (plan.stages.size()==2 && domain.first_stage==0 && domain.size()==2) {
                        // A local chain sees one token spec, never receives or emits bridge rows.
                        if (bad_prefill || !cmd.window || cmd.window->token_ids.size()!=cmd.rows) {
                            error="mock full HIP prompt refused"; return false;
                        }
                        return true;
                    }
                    if (isolated_prefill) {
                        if (bad_prefill || cmd.request_id!=2 || cmd.rows!=3 || cmd.prefill_chunk!=2 ||
                            !cmd.window || cmd.window->token_ids!=std::vector<int32_t>{1,2,3}) {
                            error="mock isolated prefill refused";return false;
                        }
                        return true; // no prefill_rows messages on either socket
                    }
                    const size_t stride=size_t(cmd.geometry.hc)*cmd.geometry.n_embd;
                    for (size_t offset=0;offset<cmd.rows;) {
                        const size_t count=std::min({size_t(prefill_piece_rows(cmd.geometry)),
                            size_t(cmd.prefill_chunk)-offset%cmd.prefill_chunk,size_t(cmd.rows)-offset});
                        if (i==0) {
                            Frame piece;
                            piece.kind=Kind::prefill_rows; piece.stage_index=cmd.stage_index;
                            piece.request_id=cmd.request_id; piece.window_id=offset+1+(bad_prefill?1:0);
                            piece.plan_fingerprint=cmd.plan_fingerprint; piece.geometry=cmd.geometry;
                            piece.rows=static_cast<uint32_t>(count);
                            piece.payload.assign(count*stride, float(offset+1));
                            send_frame(sock,piece,1200);
                        } else {
                            auto piece=receive_frame(sock,1200);
                            if (!piece || piece->kind!=Kind::prefill_rows || piece->stage_index!=i ||
                                piece->request_id!=cmd.request_id || piece->window_id!=offset+1 ||
                                piece->plan_fingerprint!=cmd.plan_fingerprint || piece->rows!=count ||
                                piece->payload!=std::vector<float>(count*stride,float(offset+1))) {
                                error="mock native residual transport mismatch"; return false;
                            }
                        }
                        offset+=count;
                    }
                    return true;
                };
                if (i + 1 == plan.stages.size() && !wrong_mode) {
                    if (delayed_startup) ::usleep(1600000);
                    MockFinal final;
                    final.bad_identity = bad;
                    final.grouped = grouped;
                    final.arm_mode = arm_mode;
                    final.mtp_mode = mtp_mode;
                    final.service_fault = service_fault;
                    final.nonfinite = nonfinite;
                    final.fail_commit = fail_commit;
                    final.unmarked_logit = unmarked_logit;
                    final.fail_reset = fail_reset;
                    serve_final_stage_worker(fd[1], plan, i, final, 1200, shared, (resident || mtp_mode || direct_verified) ? &limits : nullptr,
                                             delayed_startup ? 4000 : -1, grouped && !wrong_group_hello, pf, bind, guarded_direct);
                } else {
                    MockStageExecutor intermediate;
                    NonfiniteProducer corrupt;
                    StageExecutor& selected = bad_producer ? static_cast<StageExecutor&>(corrupt) : static_cast<StageExecutor&>(intermediate);
                    serve_worker_stateful(fd[1], plan, i, selected, 1200, shared, (resident || direct_verified) ? &limits : nullptr,
                                          delayed_startup ? 4000 : -1, pf, bind, guarded_direct);
                }
                ::close(fd[1]);
                ::_exit(0);
            } catch (...) { ::_exit(3); }
        }
        ::close(fd[1]);
        group.sockets.push_back(fd[0]);
        group.pids.push_back(pid);
    }
    return group;
}
void three_stage(bool shared, bool fanout) {
    Plan p = parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10",10,{2,3},3719);
    WorkerGroup group = launch(p, shared);
    FinalPipelineSession pipe(p, group.sockets, 1200, shared, false, fanout);
    pipe.initialize();
    const WindowSpec spec{22,{4,51,9}};
    FinalStageResult provisional = pipe.execute(input_frame(p), spec);
    check(pipe.has_pending() && !pipe.poisoned(), "result remains provisional");
    check(provisional.predicted_tokens == std::vector<int32_t>({21,68,26}),
          "final stage returned deterministic mock head predictions");
    check(provisional.stage_index == 2 && !provisional.logits, "head identity, optional logits");
    fails([&] { (void)pipe.execute(input_frame(p), spec); }, "provisional cannot be overwritten");
    check(!pipe.poisoned(), "overlap refused before touching workers");
    fails([&] { (void)pipe.commit(4); }, "oversized prefix refused");
    check(pipe.has_pending() && !pipe.poisoned(), "invalid count does not commit");
    const auto published = pipe.commit(2);
    check(published.predicted_tokens == provisional.predicted_tokens, "published only after ACK");
    check(!pipe.has_pending(), "no outstanding window after commit");
    fails([&] { (void)pipe.commit(2); }, "duplicate commit rejected");
    check(!pipe.poisoned(), "duplicate call does not corrupt pipeline");
    const auto second = pipe.execute(input_frame(p,17,2),WindowSpec{24,{9,10,11}});
    check(second.window_id == 2, "repeated pipeline window accepts increasing ID");
    check(second.logits && second.logits->size() == 3000 &&
          (*second.logits)[1] == 1.25f &&
          std::signbit((*second.logits)[0]), "optional logits transported bit-exactly");
    check(pipe.commit(0).window_id == 2, "protocol mock accepts zero prefix, GPU does not yet");
    check(pipe.execute(input_frame(p,18,1),WindowSpec{24,{2,3,4}}).window_id==1,
          "new request ID can reset window counter");
    check(pipe.commit(3).request_id == 18, "complete prefix commit on new request");
}
// Exercise only handshake/admission/replay/fail-stop on existing CPU mocks.
// No inference or numerical correctness is inferred from these assertions.
void resident_lifecycle(bool shared, bool fanout) {
    const Plan p = parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10",10,{2,3},3719);
    {
        auto w = launch(p, shared, false, false, false, false, 0.f, true);
        FinalPipelineSession pipe(p, w.sockets, 1200, shared, true, fanout);
        pipe.initialize();
        fails([&]{ (void)pipe.execute(input_frame(p,1,1),{0,{1,2,3}}); }, "reset required");
        pipe.begin_request(1);
        (void)pipe.execute(input_frame(p,1,1),{0,{1,2,3}});
        fails([&]{ pipe.begin_request(2); }, "reset cannot overwrite pending window");
        check(pipe.commit(1).request_id == 1, "positive prefix ACK");
        fails([&]{ pipe.begin_request(1); }, "replayed reset refused");
        // Resident workers may sit idle longer than a frame deadline.
        if (!shared) ::usleep(1400000);
        pipe.begin_request(2);
        fails([&]{ (void)pipe.execute(input_frame(p,1,2),{1,{1,2,3}}); }, "old request refused");
        (void)pipe.execute(input_frame(p,2,1),{0,{3,2,1}});
        check(pipe.commit(3).request_id == 2, "same workers accept new position zero");
        pipe.begin_request(3);
        (void)pipe.execute(input_frame(p,3,1),{0,{1,2,3}});
        check(pipe.commit(3).request_id == 3 && !pipe.poisoned(), "third resident request");
    }
    {
        auto w = launch(p, shared, false, false, false, false, 0.f, true, true);
        FinalPipelineSession pipe(p, w.sockets, 1200, shared, true, fanout);
        pipe.initialize();
        fails([&]{ pipe.begin_request(1); }, "later worker reset refusal");
        check(pipe.poisoned(), "partial reset poisons all peers");
        fails([&]{ pipe.begin_request(2); }, "poisoned workers not reusable");
    }
}
void grouped_domain(bool shared, bool fanout) {
    const Plan p = parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10",10,{2,3},3719);
    {
        auto w = launch(p,shared,false,false,false,false,0.f,true,false,false,true);
        check(w.pids.size()==2,"two backend children for three physical stages");
        FinalPipelineSession pipe(p,w.sockets,1200,shared,true,fanout,false,true);
        pipe.initialize(); pipe.begin_request(1);
        auto result=pipe.execute(input_frame(p,1,1),{0,{1,2,3}});
        check(result.stage_index==2 && result.plan_fingerprint==p.fingerprint(),"physical terminal identity retained");
        check(pipe.commit(1).predicted_tokens.front()==18,"one domain ACK publishes result");
        pipe.begin_request(2);
        (void)pipe.execute(input_frame(p,2,1),{0,{3,2,1}});
        check(pipe.commit(3).request_id==2,"grouped reset and new request");
    }
    {
        auto w=launch(p,shared,false,false,true,false,0.f,true,false,false,true);
        FinalPipelineSession pipe(p,w.sockets,1200,shared,true,fanout,false,true);
        pipe.initialize();pipe.begin_request(1);
        (void)pipe.execute(input_frame(p,1,1),{0,{1,2,3}});
        fails([&]{(void)pipe.commit(1);},"later group commit failure");
        check(pipe.poisoned(),"failed group cannot publish");
    }
    {
        auto w=launch(p,shared,false,false,false,false,0.f,true,false,false,true,true);
        FinalPipelineSession pipe(p,w.sockets,1200,shared,true,fanout,false,true);
        fails([&]{pipe.initialize();},"terminal-only hello must not pass grouped readiness");
    }
}
void early_window_protocol() {
    const Plan p = parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10",10,{2,3},3719);
    auto start=[&]{return launch(p,true,false,false,false,false,0.f,true,false,false,true);};
    {
        auto w=start();
        FinalPipelineSession pipe(p,w.sockets,1200,true,true,true,true,true,
                                  false,false,false,false,false,true);
        pipe.initialize(); pipe.begin_request(1);
        pipe.native_prefill({1,2},2);
        pipe.execute_discard(input_frame(p,1,2),{2,{3,4,5}});
        check(pipe.commit(2).predicted_tokens.front()==20,"early metadata native prefill and compact prefix");
        pipe.execute_discard(input_frame(p,1,3),{4,{5,6,7}});
        check(pipe.commit(3).predicted_tokens.front()==22,"early metadata continuation");
        pipe.begin_request(2);
        pipe.execute_discard(input_frame(p,2,1),{0,{1,2,3}});
        check(pipe.commit(3).request_id==2,"early metadata reset");
    }
    {
        auto w=launch(p,true,false,false,false,false,0.f,true,false,false,true,false,false,0,0,false,false,true);
        FinalPipelineSession pipe(p,w.sockets,1200,true,true,true,true,true,
                                  false,false,false,false,false,true);
        pipe.initialize(); pipe.begin_request(1);
        fails([&]{pipe.execute_discard(input_frame(p,1,1),{0,{1,2,3}});},
              "nonfinite producer rejected despite early downstream admission");
        check(pipe.poisoned(),"early metadata failure poisons all peers");
        fails([&]{(void)pipe.commit(1);},"early metadata failure cannot commit");
        fails([&]{pipe.begin_request(2);},"early metadata failure cannot reuse session");
    }
    {
        auto w=start();
        fails([&]{FinalPipelineSession pipe(p,w.sockets,1200,false,true,true,true,true,
                                           false,false,false,false,false,true);},
              "early metadata requires shared transport");
        fails([&]{FinalPipelineSession pipe(p,w.sockets,1200,true,true,true,true,true,
                                           true,false,false,false,false,true);},
              "early metadata is independent of prearmed GPU receiver");
    }
}
void hip_only_protocol(bool shared, bool grouped, bool fanout) {
    const Plan p=parse_plan("hip:0:0-5,hip:1:5-10",10,{2,3},3719);
    auto w=launch(p,shared,false,false,false,false,0.f,true,false,false,grouped);
    check(w.pids.size()==(grouped?1u:2u),"HIP-only domain process count");
    FinalPipelineSession pipe(p,w.sockets,1200,shared,true,fanout,false,grouped);
    pipe.initialize(); pipe.begin_request(1);
    if (grouped) {
        fails([&]{pipe.native_prefill({1,2,3},2,true);},"local chain excludes isolated bridge diagnostics");
        pipe.native_prefill({1,2,3,4,5},2);
        fails([&]{pipe.native_prefill({1},1);},"local prefix replay rejected");
        (void)pipe.execute(input_frame(p,1,2),{5,{6,7,8}});
    } else {
        fails([&]{pipe.native_prefill({1,2,3},2);},"separate HIP workers cannot run local bulk");
        (void)pipe.execute(input_frame(p,1,1),{0,{1,2,3}});
    }
    check(pipe.commit(1).stage_index==1,"HIP-only terminal physical identity");
    pipe.begin_request(2);
    (void)pipe.execute(input_frame(p,2,1),{0,{3,2,1}});
    check(pipe.commit(3).request_id==2,"HIP-only resident reset");
}
void hip_only_local_failure(bool shared) {
    const Plan p=parse_plan("hip:0:0-5,hip:1:5-10",10,{2,3},3719);
    auto w=launch(p,shared,false,false,false,false,0.f,true,false,false,true,false,true);
    FinalPipelineSession pipe(p,w.sockets,1200,shared,true,true,false,true);
    pipe.initialize(); pipe.begin_request(1);
    fails([&]{pipe.native_prefill({1,2,3},2);},"local prefill failure not ACKed");
    check(pipe.poisoned(),"local prefill failure poisons coordinator");
    fails([&]{pipe.begin_request(2);},"poisoned local chain is not reusable");
}
void receiver_arm_protocol(bool shared) {
    const Plan p=parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10",10,{2,3},3719);
    for (int mode : {1,2,0}) {
        auto w=launch(p,shared,false,false,false,false,0.f,true,false,false,true,false,false,mode);
        FinalPipelineSession pipe(p,w.sockets,1200,shared,true,true,false,true,true);
        pipe.initialize(); pipe.begin_request(1);
        fails([&]{pipe.native_prefill({1,2},1);},"receiver-arm rejects bulk prefill");
        if (mode!=1) {
            fails([&]{(void)pipe.execute(input_frame(p,1,1),{0,{1,2,3}});},
                  "refused or missing arm ACK cannot produce a head result");
            check(pipe.poisoned(),"failed arm poisons pipeline");
            continue;
        }
        (void)pipe.execute(input_frame(p,1,1),{0,{1,2,3}});
        check(pipe.commit(1).predicted_tokens.front()==18,"armed window retains accepted-prefix gate");
        (void)pipe.execute(input_frame(p,1,2),{1,{2,3,4}});
        check(pipe.commit(2).window_id==2,"rearm after accepted-prefix commit");
        pipe.begin_request(2);
        (void)pipe.execute(input_frame(p,2,1),{0,{3,2,1}});
        check(pipe.commit(3).request_id==2,"armed pipeline reset");
    }
    // A receiver-arm ACK is an empty, bounded, nonzero-window control.
    Frame ack=input_frame(p,1,1);
    ack.kind=Kind::window_armed; ack.stage_index=2; ack.payload.clear();
    check(decode(encode(ack)).kind==Kind::window_armed,"receiver arm ACK roundtrip");
    ack.window_id=0;
    fails([&]{(void)encode(ack);},"zero window arm ACK rejected");
}
void mtp_relay_protocol(bool shared) {
    const Plan p=parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10",10,{2,3},3719);
    for(int mode:{1,2,3,4,0}) {
        auto w=launch(p,shared,false,false,false,false,0.f,false,false,false,true,false,false,0,mode);
        FinalPipelineSession pipe(p,w.sockets,1200,shared,false,true,false,true,false,true);
        pipe.initialize();
        fails([&]{(void)pipe.request_mtp_proposal();},"proposal before commit refused");
        (void)pipe.execute(input_frame(p,1,1),{0,{1,2,3}});
        fails([&]{(void)pipe.request_mtp_proposal();},"provisional-only proposal refused");
        check(pipe.commit(2).predicted_tokens[1]==19,"anchor comes from kept row");
        if(mode!=1) {
            fails([&]{(void)pipe.request_mtp_proposal();},"refused or invalid proposal");
            check(pipe.poisoned(),"proposal failure poisons both peers");
            continue;
        }
        auto offer=pipe.request_mtp_proposal();
        check(offer.pos0==2 && offer.token_ids==std::vector<int32_t>({19,36,0}),"identity-bound proposal");
        fails([&]{(void)pipe.request_mtp_proposal();},"duplicate offer refused");
        auto result=pipe.execute(input_frame(p,1,2),offer);
        check(result.predicted_tokens[0]==36 && result.predicted_tokens[1]==53,"actual mock proposal evaluated");
        pipe.commit(2);
        offer=pipe.request_mtp_proposal();
        check(offer.pos0==4 && offer.token_ids.front()==53,"next offer after partial accept");
        offer.token_ids.back()=8;
        fails([&]{(void)pipe.execute(input_frame(p,1,3),offer);},"altered offer rejected");
        check(pipe.poisoned(),"altered offer poisons pipeline");
    }
    {
        auto w=launch(p,shared,false,false,false,false,0.f,false,false,false,true,false,false,0,1);
        FinalPipelineSession pipe(p,w.sockets,1200,shared,false,true,false,true,false,true);
        pipe.initialize();
        (void)pipe.execute(input_frame(p,1,1),{0,{1,2,3}});
        pipe.commit(2);
        const auto offer=pipe.request_mtp_proposal();
        (void)pipe.execute(input_frame(p,1,2),offer);
        fails([&]{(void)pipe.commit(3);},"cannot commit rejected proposal suffix");
        check(pipe.poisoned(),"wrong greedy prefix poisons before sending commit");
    }
    // Both endpoints must opt in to an anchor-only MTP response.
    for (const int variant : {0,1,2}) {
        auto w=launch(p,shared,false,false,false,false,0.f,false,false,false,true,false,false,0,variant==2?6:5);
        FinalPipelineSession pipe(p,w.sockets,1200,shared,false,true,false,true,false,true,variant!=1);
        pipe.initialize();
        (void)pipe.execute(input_frame(p,1,1),{0,{1,2,3}});
        pipe.commit(2);
        if (variant) {
            fails([&]{(void)pipe.request_mtp_proposal();},"anchor-only relay without dual opt-in refused");
            check(pipe.poisoned(),"singleton proposal refusal poisons pipeline");
            continue;
        }
        const auto anchor=pipe.request_mtp_proposal();
        check(anchor.pos0==2 && anchor.token_ids==std::vector<int32_t>{19},"anchor-only proposal identity");
        auto one=input_frame(p,1,2);
        one.rows=1; one.payload.resize(p.geometry.floats_per_row());
        (void)pipe.execute(one,anchor);
        check(pipe.commit(1).predicted_tokens[0]==36,"singleton target-authoritative commit");
    }
    Frame f=input_frame(p,1,1);
    f.kind=Kind::mtp_proposal; f.stage_index=2; f.payload.clear();
    f.window=WindowSpec{2,{19,36,0}};
    check(decode(encode(f)).window->token_ids==f.window->token_ids,"proposal codec roundtrip");
    f.rows=1; f.window->token_ids={19};
    check(decode(encode(f)).window->token_ids==f.window->token_ids,"singleton codec; admission checked by endpoints");
    f.rows=6; f.window->token_ids.resize(6);
    fails([&]{(void)encode(f);},"oversized MTP proposal rejected");
}
void mtp_resident_protocol(bool shared) {
    const Plan p=parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10",10,{2,3},3719);
    {
        auto w=launch(p,shared,false,false,false,false,0.f,true,false,false,true,false,false,0,7);
        FinalPipelineSession pipe(p,w.sockets,1200,shared,true,true,false,true,false,true,true,true);
        pipe.initialize();
        fails([&]{pipe.begin_request(2);},"resident MTP requires contiguous request IDs");
        pipe.begin_request(1);
        fails([&]{(void)pipe.request_mtp_proposal();},"no prior-request proposal at reset");
        fails([&]{pipe.native_prefill({1,2},1);},"resident MTP excludes bulk prefill");
        (void)pipe.execute(input_frame(p,1,1),{0,{1,2,3}});
        fails([&]{pipe.begin_request(2);},"resident MTP cannot reset provisional target");
        pipe.commit(2);
        const auto offer=pipe.request_mtp_proposal();
        check(offer.token_ids==std::vector<int32_t>({19,36,0}),"resident MTP proposal");
        (void)pipe.execute(input_frame(p,1,2),offer);
        check(pipe.commit(2).predicted_tokens[1]==53,"actual partial accepted prefix");
        // Terminal diagnostic not submitted: reset drops all prior relay identity.
        (void)pipe.request_mtp_proposal();
        pipe.begin_request(2);
        fails([&]{(void)pipe.request_mtp_proposal();},"unsubmitted previous offer cannot survive reset");
        fails([&]{(void)pipe.execute(input_frame(p,1,3),offer);},"stale request refused");
        (void)pipe.execute(input_frame(p,2,1),{0,{3,4,5}});
        pipe.commit(3);
        check(pipe.request_mtp_proposal().token_ids.front()==22,"request B uses own target anchor");
        pipe.begin_request(3);
        (void)pipe.execute(input_frame(p,3,1),{0,{1,2,3}});
        pipe.commit(2);
        check(pipe.request_mtp_proposal().token_ids==offer.token_ids,"request A reproduces offer");
        for(uint64_t r=4;r<=8;++r) pipe.begin_request(r);
        fails([&]{pipe.begin_request(9);},"bounded resident request cap");
    }
    {
        auto w=launch(p,shared,false,false,false,false,0.f,true,true,false,true,false,false,0,7);
        FinalPipelineSession pipe(p,w.sockets,1200,shared,true,true,false,true,false,true,true,true);
        pipe.initialize();
        fails([&]{pipe.begin_request(1);},"native reset refusal");
        check(pipe.poisoned(),"failed reset poisons all peers");
    }
    {
        auto w=launch(p,shared,false,false,false,false,0.f,true,false,false,true,false,false,0,1);
        fails([&]{FinalPipelineSession pipe(p,w.sockets,1200,shared,true,true,false,true,false,true,true);},
              "ordinary relay cannot opt into resident without separate flag");
    }
}
void native_prefill_protocol(bool shared) {
    const Plan p=parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10",10,{2,3},3719);
    Frame spec;
    spec.kind=Kind::prefill_spec; spec.stage_index=0; spec.request_id=1; spec.window_id=1;
    spec.plan_fingerprint=p.fingerprint(); spec.geometry=p.geometry; spec.rows=5;
    spec.window=WindowSpec{0,{1,2,3,4,5}}; spec.prefill_chunk=2;
    const auto round=decode(encode(spec));
    check(round.kind==Kind::prefill_spec && round.prefill_chunk==2 &&
          round.window && round.window->token_ids==spec.window->token_ids,"prefill spec round trip");
    Frame residual=spec;
    residual.kind=Kind::prefill_rows; residual.window.reset(); residual.prefill_chunk=0;
    residual.rows=2; residual.payload.assign(2*p.geometry.hc*p.geometry.n_embd,0.25f);
    check(decode(encode(residual)).payload==residual.payload,"R-only residual round trip");
    residual.kind=Kind::data;
    fails([&]{(void)encode(residual);},"R-only cannot be used as decode handoff");
    auto invalid=spec; invalid.prefill_chunk=0;
    fails([&]{(void)encode(invalid);},"zero native chunk rejected");
    std::string error;
    WindowAdmission gate(p,0,{0,64,1000,8});
    check(!gate.begin_prefill(spec,error),"reset required before native prompt");
    check(gate.reset_request(1,error) && gate.begin_prefill(spec,error),"admit native prompt");
    check(!gate.begin({0,{1}},1,2,error),"decode cannot overlap incomplete native prompt");
    check(gate.finished_prefill(error) && gate.next_position()==5,"position advanced after native drain");
    check(!gate.begin_prefill(spec,error),"native prompt replay rejected");
    check(gate.begin({5,{6}},1,2,error),"next decode starts at native prompt end");
    {
        auto w=launch(p,shared,false,false,false,false,0.f,true,false,false,true);
        FinalPipelineSession pipe(p,w.sockets,1200,shared,true,true,false,true);
        pipe.initialize(); pipe.begin_request(1);
        pipe.native_prefill({1,2,3,4,5},2);
        fails([&]{pipe.native_prefill({1},1);},"prefill replay declined by coordinator");
        (void)pipe.execute(input_frame(p,1,2),{5,{6,7,8}});
        check(pipe.commit(1).request_id==1,"native prompt to decode commit");
        pipe.begin_request(2);
        pipe.native_prefill({2,4,6},2);
        (void)pipe.execute(input_frame(p,2,2),{3,{8,9,10}});
        check(pipe.commit(3).request_id==2,"native prompt after resident reset");
    }
    {
        auto w=launch(p,shared,false,false,false,false,0.f,true,false,false,true,false,true);
        FinalPipelineSession pipe(p,w.sockets,1200,shared,true,true,false,true);
        pipe.initialize(); pipe.begin_request(1);
        fails([&]{pipe.native_prefill({1,2,3},2);},"out-of-order native residual rejected");
        check(pipe.poisoned(),"native prefill failure poisons coordinator");
    }
    const Plan bounded=parse_plan("cuda:0:0-12,hip:0:12-30,hip:1:30-48",48,{4,2560},3719);
    for (bool fail : {false,true}) {
        auto w=launch(bounded,shared,false,false,false,false,0.f,true,false,false,true,false,fail,0,0,true);
        FinalPipelineSession pipe(bounded,w.sockets,1200,shared,true,true,false,true);
        pipe.initialize();pipe.begin_request(1);
        fails([&]{pipe.native_prefill({1,2,3},2,true);},"isolated diagnostic requires request 2");
        pipe.begin_request(2);
        fails([&]{pipe.native_prefill({1,2,3},4,true);},"isolated diagnostic refuses oversized chunk");
        check(!pipe.poisoned(),"invalid isolated arguments do not start a worker command");
        if (fail) {
            fails([&]{pipe.native_prefill({1,2,3},2,true);},"isolated worker failure is not an ACK");
            check(pipe.poisoned(),"isolated failure poisons pipeline");
        } else {
            pipe.native_prefill({1,2,3},2,true); // workers send only prefill_done
            (void)pipe.execute(input_frame(bounded,2,2),{3,{4,5,6}});
            check(pipe.commit(1).request_id==2,"isolated prefill done ACKs allow only next committed window");
        }
    }
}

void bad_ack(bool fanout) {
    const Plan p = parse_plan("cuda:0:0-3",3,{2,3},3719);
    int fd[2]{};
    check(::socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0,"bad ACK socketpair");
    const pid_t pid=::fork();
    check(pid>=0,"bad ACK fork");
    if(pid==0) {
        ::close(fd[0]);
        try {
            Frame hello;
            hello.kind=Kind::hello_final; hello.stage_index=0;
            hello.request_id=1; hello.window_id=1; hello.rows=1;
            hello.geometry=p.geometry; hello.plan_fingerprint=p.fingerprint();
            hello.identity={static_cast<uint32_t>(Backend::cuda),0,0,3};
            send_frame(fd[1],hello,1200);
            auto ready=receive_frame(fd[1],1200);
            auto spec=receive_frame(fd[1],1200);
            auto data=receive_frame(fd[1],1200);
            if(!ready||!spec||!data||ready->kind!=Kind::ready||
               spec->kind!=Kind::window_spec||data->kind!=Kind::data)::_exit(2);
            FinalStageResult result;
            result.stage_index=0;result.request_id=data->request_id;
            result.window_id=data->window_id;result.plan_fingerprint=p.fingerprint();
            result.rows=3;result.vocab_size=1000;result.predicted_tokens={10,11,12};
            send_final_result(fd[1],result,1200);
            auto commit=receive_frame(fd[1],1200);
            if(!commit||commit->kind!=Kind::commit_prefix||commit->keep_count!=1)::_exit(3);
            commit->kind=Kind::committed_prefix;commit->keep_count=2;
            send_frame(fd[1],*commit,1200);
            ::close(fd[1]);::_exit(0);
        } catch (...) { ::_exit(4); }
    }
    ::close(fd[1]);
    {
        FinalPipelineSession pipe(p,{fd[0]},1200,false,false,fanout);
        pipe.initialize();
        check(pipe.execute(input_frame(p),{0,{1,2,3}}).predicted_tokens[1]==11,
              "bad ACK test receives provisional output");
        fails([&]{(void)pipe.commit(1);},"incorrect accepted-prefix ACK rejected");
        check(pipe.poisoned(),"bad ACK poisons final pipeline");
        fails([&]{(void)pipe.execute(input_frame(p,17,2),{0,{1,2,3}});},
              "poisoned final pipeline cannot be reused");
    }
    ::close(fd[0]);
    int status=0;
    check(::waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==0,
          "bad ACK peer exited cleanly");
}
void negative(bool shared, bool fanout) {
    const Plan p = parse_plan("cuda:0:0-3",3,{2,3},3719);
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
        WorkerGroup w = launch(p, shared, false, false, false, false, value);
        FinalPipelineSession pipe(p, w.sockets, 1200, shared, false, fanout);
        pipe.initialize();
        auto result = pipe.execute(input_frame(p),{0,{1,2,3}});
        check(result.nonfinite_rows.empty() && result.logits &&
              !std::isfinite(result.logits->back()), "unmarked nonfinite diagnostics transported");
        fails([&]{(void)pipe.commit(1);}, "unmarked nonfinite logits cannot commit even outside prefix");
        check(pipe.poisoned() && !pipe.has_pending(), "unmarked nonfinite poisons result");
    }
    {
        WorkerGroup w = launch(p, shared);
        PipelineSession pipe(p, w.sockets, 1200, shared);
        fails([&]{pipe.initialize();}, "activation-only coordinator rejects final handshake");
        check(pipe.poisoned(), "reverse capability mismatch poisons session");
    }
    {
        WorkerGroup w = launch(p, !shared);
        FinalPipelineSession pipe(p, w.sockets, 1200, shared, false, fanout);
        fails([&]{pipe.initialize();}, "wrong final transport mode rejected");
        check(pipe.poisoned(), "transport mismatch poisons session");
    }
    {
        WorkerGroup w = launch(p, shared, true);
        FinalPipelineSession pipe(p, w.sockets, 1200, shared, false, fanout);
        pipe.initialize();
        fails([&]{(void)pipe.execute(input_frame(p),{0,{1,2,3}});}, "wrong result identity rejected");
        check(pipe.poisoned(), "bad identity poisons session");
    }
    {
        WorkerGroup w = launch(p, shared, false, true);
        FinalPipelineSession pipe(p, w.sockets, 1200, shared, false, fanout);
        pipe.initialize();
        auto result = pipe.execute(input_frame(p),{0,{1,2,3}});
        check(result.nonfinite_rows == std::vector<uint32_t>({1}), "diagnostics available provisional");
        fails([&]{(void)pipe.commit(3);}, "nonfinite result refused before commit");
        check(pipe.poisoned(), "nonfinite poison permanently prevents publish");
    }
    {
        const Plan split = parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10",10,{2,3},3719);
        WorkerGroup w = launch(split, shared, false, false, true);
        FinalPipelineSession pipe(split, w.sockets, 1200, shared, false, fanout);
        pipe.initialize();
        (void)pipe.execute(input_frame(split),{0,{1,2,3}});
        fails([&]{(void)pipe.commit(1);}, "final failure after two stage commits not published");
        check(pipe.poisoned(), "failed final commit poisons all peers");
    }
    {
        WorkerGroup w = launch(p, shared, false, false, false, true);
        FinalPipelineSession pipe(p, w.sockets, 1200, shared, false, fanout);
        fails([&]{pipe.initialize();}, "wrong handshake fails closed");
        check(pipe.poisoned(), "wrong final capability poisons coordinator");
    }
}
void guarded_direct_protocol() {
    const Plan p = parse_plan("cuda:0:0-15,hip:0:15-33,hip:1:33-48",48,{4,2560},3719);
    auto start=[&](bool corrupt, bool guarded, int arm){
        return launch(p,true,false,false,false,false,0.f,true,false,false,true,false,false,arm,0,false,true,corrupt,guarded);
    };
    {
        auto w=start(false,true,1);
        FinalPipelineSession pipe(p,w.sockets,4000,true,true,false,false,true,
                                  false,false,false,false,true,false,true);
        pipe.initialize(); pipe.begin_request(1);
        pipe.execute_discard(input_frame(p,1,1),{0,{1,2,3}});
        check(pipe.commit(2).predicted_tokens[0]==18,"guarded direct partial prefix");
        pipe.execute_discard(input_frame(p,1,2),{2,{4,5,6}});
        check(pipe.commit(3).predicted_tokens[0]==21,"guarded direct slot reuse");
        pipe.begin_request(2);
        pipe.execute_discard(input_frame(p,2,1),{0,{1,2,3}});
        check(pipe.commit(3).request_id==2,"guarded direct resident reset");
    }
    {
        auto w=start(true,true,1);
        FinalPipelineSession pipe(p,w.sockets,4000,true,true,false,false,true,
                                  false,false,false,false,true,false,true);
        pipe.initialize(); pipe.begin_request(1);
        fails([&]{pipe.execute_discard(input_frame(p,1,1),{0,{1,2,3}});},"guarded direct corrupt producer blocked");
        check(pipe.poisoned(),"guarded direct failure poisons all peers");
        fails([&]{(void)pipe.commit(1);},"guarded direct failure cannot commit");
    }
    {
        auto w=start(false,false,0);
        FinalPipelineSession pipe(p,w.sockets,4000,true,true,false,false,true,
                                  false,false,false,false,true,false,true);
        fails([&]{pipe.initialize();},"legacy direct worker cannot attest guarded publication");
        check(pipe.poisoned(),"guarded capability mismatch poisons session");
    }
}
void direct_verified_protocol() {
    const Plan p = parse_plan("cuda:0:0-15,hip:0:15-33,hip:1:33-48",48,{4,2560},3719);
    check(direct_verified_plan(p),"preferred mixed direct admission");
    check(direct_verified_plan(parse_plan("cuda:0:0-12,hip:0:12-30,hip:1:30-48",48,{4,2560},3719)),
          "reference placement direct admission");
    check(!direct_verified_plan(parse_plan("hip:0:0-24,hip:1:24-48",48,{4,2560},3719)),
          "direct mode rejects HIP-only plan");
    auto start=[&] { return launch(p,true,false,false,false,false,0.f,true,false,false,true,false,false,0,0,false,true); };
    {
        auto w=start();
        FinalPipelineSession pipe(p,w.sockets,4000,true,true,false,false,true,false,false,false,false,true);
        pipe.initialize();
        pipe.begin_request(1);
        pipe.execute_discard(input_frame(p,1,1),{0,{1,2,3}});
        check(pipe.commit(2).predicted_tokens[0]==18,"direct descriptor handshake and partial commit");
        pipe.execute_discard(input_frame(p,1,2),{2,{4,5,6}});
        check(pipe.commit(3).predicted_tokens[0]==21,"same slot safely reused after commit");
        pipe.begin_request(2);
        pipe.execute_discard(input_frame(p,2,1),{0,{1,2,3}});
        check(pipe.commit(3).predicted_tokens[0]==18,"direct reset/reuse");
        fails([&]{ pipe.native_prefill({1,2},1); },"direct mode rejects native bulk");
    }
    {
        auto w=start();
        FinalPipelineSession pipe(p,w.sockets,4000,true,true,false,false,true,false,false,false,false,true);
        pipe.initialize();
        pipe.begin_request(1);
        auto bad=input_frame(p,1,1);
        bad.payload.back()=std::numeric_limits<float>::quiet_NaN();
        fails([&]{ pipe.execute_discard(std::move(bad),{0,{1,2,3}}); },"direct shared finite gate rejects malformed input");
        check(pipe.poisoned(),"nonfinite activation poisons direct session");
    }
    {
        auto w=launch(p,true,false,false,false,false,0.f,true,false,false,true,false,false,0,0,false,true,true);
        FinalPipelineSession pipe(p,w.sockets,4000,true,true,false,false,true,false,false,false,false,true);
        pipe.initialize();
        pipe.begin_request(1);
        fails([&]{ pipe.execute_discard(input_frame(p,1,1),{0,{1,2,3}}); },"direct producer nonfinite output rejected before consumer admission");
        check(pipe.poisoned(),"nonfinite producer output poisons direct session");
        fails([&]{ (void)pipe.commit(1); },"nonfinite producer output cannot commit");
    }
    {
        auto w=launch(p,true,false,false,false,false,0.f,true,false,false,true);
        FinalPipelineSession pipe(p,w.sockets,4000,true,true,false,false,true,false,false,false,false,true);
        fails([&]{ pipe.initialize(); },"ordinary shared worker cannot impersonate direct registration");
        check(pipe.poisoned(),"capability mismatch poisons direct session");
    }
}

void startup_barrier(bool shared) {
    const Plan p = parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10",10,{2,3},3719);
    auto w=launch(p,shared,false,false,false,false,0.f,true,false,true);
    FinalPipelineSession pipe(p,w.sockets,4000,shared,true,true);
    pipe.initialize(); // later final worker's load exceeds early worker I/O budget
    pipe.begin_request(1);
    pipe.execute_discard(input_frame(p,1,1),{0,{1,2,3}});
    check(pipe.commit(3).predicted_tokens[0]==18,"slow peer readiness does not expire prepared worker frame deadline");
}
void startup_frame_failures() {
    int fd[2]{};
    check(::socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0,"startup test socketpair");
    fails([&]{ (void)receive_startup_frame(fd[0],20,30); },"startup idle wait bounded");
    const unsigned char partial=0;
    check(::write(fd[1],&partial,1)==1,"write partial startup frame");
    fails([&]{ (void)receive_startup_frame(fd[0],20,1000); },"partial READY retains frame deadline");
    ::close(fd[0]); ::close(fd[1]);
    check(::socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0,"startup EOF socketpair");
    ::close(fd[1]);
    check(!receive_startup_frame(fd[0],20,1000),"supervisor EOF releases prepared worker");
    ::close(fd[0]);
}
void mtp_service_protocol(bool hip_only, bool optimized=false) {
    const auto p=parse_plan(hip_only ? "hip:0:0-5,hip:1:5-10" : "cuda:0:0-3,hip:0:3-7,hip:1:7-10",10,{2,3},3719);
    auto start=[&](int mode){return launch(p,true,false,false,false,false,0.f,true,false,false,true,false,false,0,mode);};
    const MtpRequestPolicy a{2048,8,4,248320,{1,2,3},{99}};
    {
        auto w=start(8);
        FinalPipelineSession pipe(p,w.sockets,1200,true,true,optimized,optimized,true,false,true,true,true,false,optimized&&!hip_only,false,true);
        pipe.initialize(); pipe.begin_request(1); pipe.configure_mtp_request(1,a);
        fails([&]{pipe.configure_mtp_request(1,a);},"duplicate MTP request metadata refused");
        check(!pipe.poisoned(),"preflight duplicate metadata does not start work");
        pipe.execute_discard(input_frame(p,1,1),{0,{1,2,3}});
        check(pipe.commit(3).predicted_tokens.back()==20,"MTP service prompt commit");
        auto offer=pipe.request_mtp_proposal();
        check(offer.token_ids==std::vector<int32_t>({20,37,0}),"service offer bound to actual target");
        pipe.execute_discard(input_frame(p,1,2),offer);
        check(pipe.commit(2).predicted_tokens[1]==54,"service partial accepted prefix");
        pipe.begin_request(2);
        const MtpRequestPolicy b{2048,1,4,248320,{7,8,9},{99}};
        pipe.configure_mtp_request(2,b);
        pipe.execute_discard(input_frame(p,2,1),{0,{7,8,9}});
        check(pipe.commit(3).request_id==2,"variable request budget after reset");
        fails([&]{(void)pipe.request_mtp_proposal();},"terminal service offer refused");
        check(pipe.poisoned(),"invalid speculative attempt poisons session");
    }
    if (optimized) {
        auto w=start(8);
        FinalPipelineSession pipe(p,w.sockets,1200,true,true,true,true,true,false,true,true,true,false,!hip_only,false,true);
        pipe.initialize(); pipe.begin_request(1);
        const MtpRequestPolicy bad{2048,8,4,248320,{9,2,3},{99}};
        pipe.configure_mtp_request(1,bad);
        fails([&]{pipe.execute_discard(input_frame(p,1,1),{0,{9,2,3}});},
              "compact MTP rejects full-logit reply");
        check(pipe.poisoned(),"unexpected diagnostic logits poison compact service");
    }
    if (optimized) {
        auto w=launch(p,true,false,true,false,false,0.f,true,false,false,true,false,false,0,8);
        FinalPipelineSession pipe(p,w.sockets,1200,true,true,true,true,true,false,true,true,true,false,!hip_only,false,true);
        pipe.initialize(); pipe.begin_request(1); pipe.configure_mtp_request(1,a);
        pipe.execute_discard(input_frame(p,1,1),{0,{1,2,3}});
        fails([&]{(void)pipe.commit(3);},"compact MTP nonfinite refuses all prefix commits");
        check(pipe.poisoned(),"compact MTP nonfinite poisons session");
    }
    {
        auto w=start(7);
        FinalPipelineSession pipe(p,w.sockets,1200,true,true,optimized,optimized,true,false,true,true,true,false,optimized&&!hip_only,false,true);
        fails([&]{pipe.initialize();},"legacy resident MTP cannot advertise service capability");
        check(pipe.poisoned(),"stale service handshake poisons session");
    }
    {
        auto w=start(8);
        FinalPipelineSession pipe(p,w.sockets,1200,true,true,optimized,optimized,true,false,true,true,true,false,optimized&&!hip_only,false,true);
        pipe.initialize(); pipe.begin_request(1); pipe.configure_mtp_request(1,a);
        fails([&]{pipe.execute_discard(input_frame(p,1,1),{0,{1,2,4}});},"service cannot change admitted prompt");
        check(pipe.poisoned(),"wrong prompt poisons service");
    }
}
// CPU-only: actual socket transport of the largest admitted MTP metadata,
 // explicit total deadline, then cancellation checkpoint/reset on the same
 // grouped endpoint. The mock never allocates GPU state or evaluates tokens.
void mtp_long_context_protocol() {
    const Plan p=parse_plan("hip:0:0-5,hip:1:5-10",10,{2,3},3719);
    auto w=launch(p,true,false,false,false,false,0.f,true,false,false,true,false,false,0,8,
                  false,false,false,false,262144);
    FinalPipelineSession pipe(p,w.sockets,5000,true,true,false,false,true,false,true,true,true,false,false,false,true);
    pipe.initialize(); pipe.begin_request(1);
    const MtpRequestPolicy long_req{262144,1,4,248320,std::vector<int32_t>(262143,17),{99}};
    pipe.configure_mtp_request(1,long_req);
    const std::vector<int32_t> prefix(262142,17);
    fails([&]{pipe.native_prefill(prefix,32);},"extended MTP requires explicit total deadline");
    check(!pipe.poisoned(),"missing deadline rejected before worker execution");
    pipe.native_prefill(prefix,32,false,5000);
    // No final verifier call for request 1: reset after the bulk checkpoint.
    pipe.begin_request(2);
    const MtpRequestPolicy short_req{262144,1,4,248320,{7,8,9},{99}};
    pipe.configure_mtp_request(2,short_req);
    pipe.native_prefill({7,8},32,false,5000);
    auto frame=input_frame(p,2,2);
    frame.rows=1; frame.payload.resize(p.geometry.floats_per_row());
    pipe.execute_discard(frame,{2,{9}});
    check(pipe.commit(1).request_id==2,"long-context MTP reset and short recovery");
}
// CPU-only partial failure retirement: failure frames preserve the cause,
// no failed checkpoint can publish, and the final worker exits nonzero.
// The mixed case includes another stage which may already have committed.
void mtp_failure_retirement(bool hip_only) {
    const Plan p=parse_plan(hip_only ? "hip:0:0-5,hip:1:5-10" :
        "cuda:0:0-3,hip:0:3-7,hip:1:7-10",10,{2,3},3719);
    const MtpRequestPolicy policy{2048,8,4,248320,{1,2,3},{99}};
    for (int fault=1;fault<=5;++fault) {
        // 1 metadata, 2 proposal, 3 observer after target commit,
        // 4 prefill handler, 5 reset.
        auto w=launch(p,true,false,false,false,false,0.f,true,fault==5,false,true,false,
                      fault==4,0,8,false,false,false,false,64,fault);
        FinalPipelineSession pipe(p,w.sockets,1200,true,true,true,false,true,false,true,true,true,false,false,false,true);
        pipe.initialize();
        std::string error;
        try {
            pipe.begin_request(1);
            pipe.configure_mtp_request(1,policy);
            if (fault==4) pipe.native_prefill({1,2},2);
            else {
                pipe.execute_discard(input_frame(p,1,1),{0,{1,2,3}});
                (void)pipe.commit(3);
                if (fault==2) (void)pipe.request_mtp_proposal();
            }
        } catch (const std::exception& e) { error=e.what(); }
        const char* cause=fault==1?"CPU service metadata refusal":
            fault==2?"CPU proposal refusal":fault==3?"CPU observer refused after target commit":
            fault==4?(hip_only?"mock full HIP prompt refused":"native prefill residual identity, order or finite gate failed"):
            "CPU protocol reset refused";
        check(error.find(cause)!=std::string::npos,"MTP failure carries source cause");
        check(pipe.poisoned() && !pipe.has_pending(),"failed MTP service retired");
        fails([&]{pipe.begin_request(2);},"retired MTP service cannot reset");
        fails([&]{(void)pipe.commit(1);},"retired MTP service cannot publish");
        int status=0; pid_t result;
        do { result=::waitpid(w.pids.back(),&status,0); } while(result<0 && errno==EINTR);
        check(result==w.pids.back() && WIFEXITED(status) && WEXITSTATUS(status)!=0,
              "failed MTP terminal process exits nonzero");
        w.pids.pop_back(); // already reaped
    }
}
void mtp_zero_tail_commit_retirement() {
    const Plan p=parse_plan("hip:0:0-5,hip:1:5-10",10,{2,3},3719);
    auto w=launch(p,true,false,false,false,false,0.f,true,false,false,true,false,false,0,8,
                  false,false,false,false,2048);
    FinalPipelineSession pipe(p,w.sockets,1200,true,true,false,false,true,false,true,true,true,false,false,false,true);
    pipe.initialize(); pipe.begin_request(1);
    const MtpRequestPolicy policy{2048,4,4,248320,std::vector<int32_t>(2044,10),{99}};
    pipe.configure_mtp_request(1,policy);
    pipe.native_prefill(std::vector<int32_t>(2043,10),32);
    auto frame=input_frame(p,1,2);
    frame.rows=1; frame.payload.resize(p.geometry.floats_per_row());
    pipe.execute_discard(frame,{2043,{10}});
    const auto first=pipe.commit(1);
    frame.window_id=3;
    pipe.execute_discard(frame,{2044,{first.predicted_tokens.front()}});
    std::string error;
    try { (void)pipe.commit(0); } catch(const std::exception& e) {error=e.what();}
    check(error.find("positive accepted prefix")!=std::string::npos,
          "zero keep forbidden before sending a tail commit");
    check(pipe.poisoned() && !pipe.has_pending(),"zero keep retires service");
    fails([&]{pipe.begin_request(2);},"retired tail cannot reset");
    int status=0; pid_t result;
    do { result=::waitpid(w.pids.back(),&status,0); } while(result<0 && errno==EINTR);
    check(result==w.pids.back() && WIFEXITED(status) && WEXITSTATUS(status)!=0,
          "EOF with a provisional MTP window is nonzero retirement");
    w.pids.pop_back();
}
void compact_and_prefix(bool shared, bool fanout) {
    const Plan p = parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10",10,{2,3},3719);
    {
        auto w=launch(p,shared);
        FinalPipelineSession pipe(p,w.sockets,1200,shared,false,fanout,true);
        pipe.initialize();
        pipe.execute_discard(input_frame(p), {0,{1,2,3}});
        const auto result=pipe.commit(1);
        check(!result.logits && result.predicted_tokens[0]==18,"compact discard path commits");
        fails([&]{ pipe.execute_discard(input_frame(p,17,2),{1,{9,10,11}}); },
              "compact request rejects unexpected diagnostic logits");
        check(pipe.poisoned(),"unexpected logits poison compact session");
    }
    check(greedy_accepted_prefix({1,2,3},{2,3,4})==3,"all draft rows accepted");
    check(greedy_accepted_prefix({1,2,3},{2,8,4})==2,"middle mismatch terminates prefix");
    check(greedy_accepted_prefix({1,2,3},{8,3,4})==1,"first mismatch keeps only anchor");
    // Report 0017: actual native MTP proposal and target-verifier replay.
    check(greedy_accepted_prefix({264,14075,8818,364,264},{2724,8818,11176,264,491})==1,
          "real MTP continuation rejects first draft and keeps anchor");
    check(greedy_accepted_prefix({1},{2})==1,"plain decode prefix");
    fails([]{ (void)greedy_accepted_prefix({1,2},{3}); },"wrong shape");
    fails([]{ (void)greedy_accepted_prefix({1,2},{2,-1}); },"negative predictions");
}
} // namespace
int main() {
    try {
        startup_barrier(false); startup_barrier(true);
        startup_frame_failures();
        direct_verified_protocol();
        guarded_direct_protocol();
        early_window_protocol();
        native_prefill_protocol(false); native_prefill_protocol(true);
        hip_only_local_failure(false); hip_only_local_failure(true);
        receiver_arm_protocol(false); receiver_arm_protocol(true);
        mtp_relay_protocol(false); mtp_relay_protocol(true);
        mtp_resident_protocol(false); mtp_resident_protocol(true);
        mtp_service_protocol(false); mtp_service_protocol(true);
        mtp_service_protocol(false,true); mtp_service_protocol(true,true);
        mtp_long_context_protocol();
        mtp_failure_retirement(false); mtp_failure_retirement(true);
        mtp_zero_tail_commit_retirement();
        for (bool fanout : {false,true}) {
            three_stage(false,fanout); three_stage(true,fanout);
            resident_lifecycle(false,fanout); resident_lifecycle(true,fanout);
            negative(false,fanout); negative(true,fanout);
            compact_and_prefix(false,fanout); compact_and_prefix(true,fanout);
            grouped_domain(false,fanout); grouped_domain(true,fanout);
            for (bool grouped : {false,true}) {
                hip_only_protocol(false,grouped,fanout); hip_only_protocol(true,grouped,fanout);
            }
            bad_ack(fanout);
        }
        std::cout << "final pipeline tests: PASS (" << assertions
                  << " assertions; isolated CPU workers, no GPU)\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "final pipeline tests: FAIL: " << ex.what() << '\n';
        return 1;
    }
}
