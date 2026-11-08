// M1.18: real AF_UNIX v4 protocol, with the production server-side window
// admission gate but a CPU-only model executor. No GPU runtime is linked.
#include "strata_heterogeneous/host.hpp"
#include "strata_heterogeneous/window_admission.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace strata::heterogeneous;
namespace {
int assertions=0;
void check(bool condition,const char* label) {
    ++assertions;
    if(!condition)throw std::runtime_error(label);
}
class CpuWindowStage final : public StageExecutor {
public:
    explicit CpuWindowStage(bool bad_output=false) : bad_output_(bad_output) {}
    bool prepare(const Plan&,size_t,std::string&) override { return true; }
    bool begin_window(const WindowSpec& w,uint64_t,uint64_t,std::string&) override {
        if(pending_||begun_)return false;
        begun_=true;rows_=static_cast<uint32_t>(w.token_ids.size());return true;
    }
    bool run_window(const Frame& in,Frame& out,std::string&) override {
        if(!begun_||pending_||in.rows!=rows_)return false;
        out=in;
        for(float& x:out.payload)x+=0.125f;
        if(bad_output_)out.payload[0]=std::numeric_limits<float>::quiet_NaN();
        pending_=true;return true;
    }
    bool commit_prefix(uint64_t,uint64_t,uint32_t keep,std::string&) override {
        if(!pending_||keep==0||keep>rows_)return false;
        pending_=begun_=false;return true;
    }
    bool commit_window(uint64_t,uint64_t,std::string&) override {return false;}
    void abandon_window() noexcept override {pending_=begun_=false;}
    void shutdown() noexcept override {pending_=begun_=false;}
private:
    bool bad_output_=false,pending_=false,begun_=false;
    uint32_t rows_=0;
};
Plan make_plan() {return parse_plan("cuda:0:0-4",4,{2,3},20261108);}
Frame input(const Plan& plan,uint64_t window) {
    Frame frame;
    frame.kind=Kind::data;frame.request_id=23;frame.window_id=window;
    frame.plan_fingerprint=plan.fingerprint();frame.geometry=plan.geometry;
    frame.rows=2;frame.payload.assign(2*plan.geometry.floats_per_row(),0.25f);
    return frame;
}
WindowSpec spec(int64_t pos) {return {pos,{1,2}};}
void scenario(bool shared, int failure) {
    const Plan plan=make_plan();
    const WindowLimits limits{0,16,100,8};
    int fds[2]{};
    check(::socketpair(AF_UNIX,SOCK_STREAM,0,fds)==0,"socketpair");
    const pid_t pid=::fork();check(pid>=0,"fork");
    if(pid==0) {
        ::close(fds[0]);
        try {
            CpuWindowStage executor(failure==3);
            serve_worker_stateful(fds[1],plan,0,executor,1000,shared,&limits);
            ::close(fds[1]);::_exit(0);
        } catch(const std::exception& ex) {
            std::cerr<<"admission worker test: "<<ex.what()<<'\n';
            ::_exit(2);
        }
    }
    ::close(fds[1]);
    try {
        PipelineSession session(plan,{fds[0]},1000,shared);
        session.initialize();
        Frame result=session.execute_window(input(plan,1),spec(0));
        if(failure==3) {
            check(false,"nonfinite output must be rejected");
        } else {
            check(result.rows==2,"provisional rows transferred");
            if(failure==1) {
                bool rejected=false;
                try {(void)session.commit_pending(0);}catch(const std::exception&){rejected=true;}
                check(rejected&&session.poisoned(),"zero-prefix commit poisons coordinator");
            } else {
                Frame accepted=session.commit_pending(1);
                check(accepted.rows==2&&!session.poisoned(),"positive prefix ACK after CPU stage");
                bool rejected=false;
                try {
                    (void)session.execute_window(input(plan,2),spec(failure==2?0:1));
                } catch(const std::exception&){rejected=true;}
                if(failure==2)
                    check(rejected&&session.poisoned(),"replayed position rejected by worker");
                else {
                    check(!rejected,"following window admitted at accepted position");
                    (void)session.commit_pending(2);
                    check(!session.poisoned(),"full accepted prefix acknowledged");
                }
            }
        }
    } catch(const std::exception&) {
        if(failure!=3)throw;
        check(true,"nonfinite output failed before publication");
    }
    ::close(fds[0]);
    int status=0;
    check(::waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==0,
          "fail-stop worker exited normally after closing protocol");
}
} // namespace
int main() {
    try {
        for(bool shared : {false,true})for(int failure=0;failure<=3;++failure)scenario(shared,failure);
        std::cout<<"M1.18 worker protocol gate: PASS ("<<assertions
                 <<" assertions, real forked CPU-only workers)\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"M1.18 worker protocol gate: FAIL: "<<e.what()<<'\n';return 1;
    }
}
