// CPU-only final-worker protocol. Real forked worker processes, no GPU model.
#include "strata_heterogeneous/final_pipeline.hpp"
#include "strata_heterogeneous/backend_domains.hpp"
#include "strata_heterogeneous/speculation.hpp"
#include "strata_heterogeneous/shared.hpp"
#include "strata_heterogeneous/window_admission.hpp"
#include <iostream>
#include <cmath>
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
        return mock_.commit_prefix(r,w,k,e);
    }
    void abandon_window() noexcept override { mock_.abandon_window(); }
    void shutdown() noexcept override { mock_.shutdown(); }
private:
    MockStageExecutor mock_;
    Plan plan_;
    uint32_t index_ = 0;
    std::vector<int32_t> tokens_;
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
                   bool delayed_startup = false, bool grouped = false, bool wrong_group_hello = false) {
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
                const WindowLimits limits{0, 64, 1000, 8};
                if (i + 1 == plan.stages.size() && !wrong_mode) {
                    if (delayed_startup) ::usleep(1600000);
                    MockFinal final;
                    final.bad_identity = bad;
                    final.grouped = grouped;
                    final.nonfinite = nonfinite;
                    final.fail_commit = fail_commit;
                    final.unmarked_logit = unmarked_logit;
                    final.fail_reset = fail_reset;
                    serve_final_stage_worker(fd[1], plan, i, final, 1200, shared, resident ? &limits : nullptr,
                                             delayed_startup ? 4000 : -1, grouped && !wrong_group_hello);
                } else {
                    MockStageExecutor intermediate;
                    serve_worker_stateful(fd[1], plan, i, intermediate, 1200, shared, resident ? &limits : nullptr,
                                          delayed_startup ? 4000 : -1);
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
    check(greedy_accepted_prefix({1},{2})==1,"plain decode prefix");
    fails([]{ (void)greedy_accepted_prefix({1,2},{3}); },"wrong shape");
    fails([]{ (void)greedy_accepted_prefix({1,2},{2,-1}); },"negative predictions");
}
} // namespace
int main() {
    try {
        startup_barrier(false); startup_barrier(true);
        startup_frame_failures();
        for (bool fanout : {false,true}) {
            three_stage(false,fanout); three_stage(true,fanout);
            resident_lifecycle(false,fanout); resident_lifecycle(true,fanout);
            negative(false,fanout); negative(true,fanout);
            compact_and_prefix(false,fanout); compact_and_prefix(true,fanout);
            grouped_domain(false,fanout); grouped_domain(true,fanout);
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
