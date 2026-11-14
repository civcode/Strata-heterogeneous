// Same final-CUDA adapter source compiled against CPU CUDA/Verifier doubles.
// No real runtime, weights, model loading or numerical GPU assertions.
#include "strata_heterogeneous/hip_final_verifier.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::heterogeneous;
namespace fv = strata::core::fake_final;
namespace {
unsigned assertions = 0;
void check(bool ok, const char* what) {
    ++assertions;
    if (!ok) throw std::runtime_error(what);
}
void pool(void* user, const float*, const int32_t*, int64_t, int64_t, float*, int64_t) {
    ++*static_cast<int*>(user);
}
struct Fixture {
    strata::core::WeightTable weights;
    strata::core::ModelGeometry model;
    strata::core::SessionState session;
    int pools = 0;
    Plan plan;
    Fixture() {
        plan.total_layers = 4;
        plan.model_fingerprint = 123;
        plan.geometry = {2,3};
        plan.stages = {{Backend::hip, 0, 0, 2}, {Backend::hip, 1, 2, 3}, {Backend::hip, 2, 3, 4}};
    }
    HipVerifierBindings bindings(int64_t initial=0, int timeout=4) {
        return {weights, model, session, {}, pool, &pools, 123, 8, timeout, initial};
    }
    Frame frame(uint32_t rows=2, uint64_t request=1, uint64_t window=1) {
        Frame f;
        f.stage_index=2;
        f.request_id=request; f.window_id=window;
        f.plan_fingerprint=plan.fingerprint();
        f.rows=rows; f.geometry=plan.geometry;
        return f;
    }
};
struct Buffer {
    std::vector<float> values;
    ActivationView view;
    Buffer(uint32_t rows, Geometry geometry)
        : values(rows*geometry.floats_per_row(),3.f),
          view(reinterpret_cast<uint8_t*>(values.data()),values.size()*4,geometry,rows) {}
};
void success_and_replay() {
    fv::reset();
    Fixture f;
    {
        auto bindings=f.bindings(9);
        HipFinalVerifierExecutor stage(bindings);
        std::string err;
        check(stage.prepare(f.plan,2,err),"prepared loaded last-stage verifier");
        check(fv::publish && fv::sampling && fv::begin==3 && fv::end==4,
              "final verifier configured for deferred commit and head sampling");
        check(fv::input!=nullptr && fv::output==nullptr && fake_cuda::allocations==2,
              "final stage owns input plus pinned host; no activation output buffer");
        const WindowSpec spec{9,{5,6}};
        check(stage.begin_window(spec,1,1,err),"begins verifier window");
        Buffer buffer(2,f.plan.geometry);
        FinalStageResult result;
        const int copies=fake_cuda::copies;
        check(stage.run_final_window(f.frame(),buffer.view,result,err),"first verifier final window");
        check(fake_cuda::copies==copies+1,"one explicit host-to-device activation copy");
        check(fv::pos==9 && fv::tokens==std::vector<int32_t>({5,6}) && f.pools==1,
              "real verifier receives tokens, position and pool callback");
        check(result.rows==2 && result.vocab_size==100 && result.stage_index==2 &&
              result.request_id==1 && result.window_id==1 &&
              result.predicted_tokens==std::vector<int32_t>({6,7}),
              "final-result identity and actual head predictions preserved");
        check(result.logits.has_value() && result.logits->size()==200 &&
              (*result.logits)[0]==0.f && (*result.logits)[199]==1.99f,
              "optional logits read row-by-row from verifier");
        check(!result.has_nonfinite() && fv::logits_copies==2,"clean diagnostic result");
        check(stage.commit_prefix(1,1,1,err) && fv::keep==1 && fv::commits==1,
              "deferred positive-prefix commit once");
        check(!stage.begin_window({9,{7}},1,2,err),"stale position rejected");
        check(stage.begin_window({10,{7}},1,2,err),"next window advanced by accepted prefix");
        Buffer one(1,f.plan.geometry);
        check(stage.run_final_window(f.frame(1,1,2),one.view,result,err),"second window");
        const int commits=fv::commits;
        check(!stage.commit_prefix(1,2,0,err) && fv::commits==commits,
              "zero-prefix refuses to fake state restoration");
        stage.abandon_window();
        stage.shutdown();
    }
    check(fake_cuda::allocations==0,"all CUDA and pinned buffers released");
}
void invalid_inputs() {
    fv::reset();Fixture f;std::string err;
    {
        auto b=f.bindings(); HipFinalVerifierExecutor stage(b);
        check(stage.prepare(f.plan,2,err),"ready for validation tests");
        check(!stage.begin_window({0,{100}},1,1,err),"vocab checked");
        check(!stage.begin_window({-1,{1}},1,1,err),"position checked");
        check(!stage.begin_window({32,{1}},1,1,err),"context boundary checked");
        check(!stage.begin_window({0,{1,2,3,4,5,6,7,8,9}},1,1,err),"max_t enforced");
        check(stage.begin_window({0,{1,2}},1,1,err),"valid context after refusals");
        check(!stage.begin_window({0,{1}},1,2,err),"overlapping context refused");
        Buffer input(2,f.plan.geometry); FinalStageResult output;
        Frame wrong=f.frame();wrong.window_id=2;
        check(!stage.run_final_window(wrong,input.view,output,err),"wrong window ID refused");
        wrong=f.frame();wrong.plan_fingerprint++;
        check(!stage.run_final_window(wrong,input.view,output,err),"wrong model plan refused");
        wrong=f.frame();wrong.stage_index=1;
        check(!stage.run_final_window(wrong,input.view,output,err),"wrong stage refused");
        check(!stage.commit_prefix(1,1,1,err),"commit before successful run refused");
        check(!stage.run_window(f.frame(),wrong,err),"final stage never sends activation frame");
        check(!stage.commit_window(1,1,err),"legacy commit refused");
    }
    check(fake_cuda::allocations==0,"validation failure cleanup");
    for (int test=0;test<5;++test) {
        auto plan=f.plan; auto bindings=f.bindings();
        if(test==0)plan.stages[2].backend=Backend::cuda;
        if(test==1)plan.stages[2].end_layer=3;
        if(test==2)bindings.model_fingerprint=999;
        if(test==3)f.session.layer_lo=2;
        if(test==4)bindings.pool=nullptr;
        HipFinalVerifierExecutor stage(bindings);
        check(!stage.prepare(plan,2,err),"bad model, backend or stage carve rejected");
        check(fake_cuda::allocations==0,"bad model rejected before allocation");
        f.session.layer_lo=3;
    }
}
void nonfinite_fail_closed() {
    fv::reset();Fixture f;std::string err;
    {
        auto b=f.bindings(); HipFinalVerifierExecutor stage(b);
        check(stage.prepare(f.plan,2,err),"nonfinite test prepared");
        check(stage.begin_window({0,{1}},1,1,err),"nonfinite window begun");
        Buffer in(1,f.plan.geometry); FinalStageResult result;
        fv::nan_logit=true; // worker's nonfinite marker is *absent*
        check(stage.run_final_window(f.frame(1),in.view,result,err),"nonfinite result sent provisionally");
        check(result.nonfinite_rows.empty() && result.has_nonfinite(),
              "unmarked NaN detected through downloaded logits");
        const int commits=fv::commits;
        check(!stage.commit_prefix(1,1,1,err) && fv::commits==commits,
              "unmarked NaN can never commit even if caller ignores result gate");
    }
    check(fake_cuda::allocations==0,"nonfinite cleanup");
    fv::reset();
    {
        auto b=f.bindings(); HipFinalVerifierExecutor no_logits(b,{false});
        check(!no_logits.prepare(f.plan,2,err),"head without logits must have enabled GPU nonfinite guard");
        fv::guard_on=true;
        check(no_logits.prepare(f.plan,2,err),"nonfinite guard permits omit logits");
        check(no_logits.begin_window({0,{1}},1,1,err),"guarded window begun");
        Buffer in(1,f.plan.geometry);FinalStageResult result;
        fv::nonfinite=true;
        check(no_logits.run_final_window(f.frame(1),in.view,result,err),"guarded window provisional");
        check(!result.logits && result.nonfinite_rows==std::vector<uint32_t>({0}),
              "GPU nonfinite row diagnostic carried without logits");
        check(!no_logits.commit_prefix(1,1,1,err),"guarded nonfinite result cannot commit");
    }
    check(fake_cuda::allocations==0,"guard cleanup");
}
void grouped_native_chain() {
    fv::reset(); Fixture f; std::string err;
    f.plan.stages = {{Backend::cuda,0,0,2},{Backend::hip,0,2,3},{Backend::hip,1,3,4}};
    strata::core::SessionState front{2,3,32};
    int pools0=0, pools1=0;
    int32_t r0[8]={-1,-1,-1,-1,0,1,-1,-1};
    int32_t r1[8]={-1,-1,-1,-1,-1,-1,0,1};
    uint8_t cache=1;
    const auto zero=[](strata::core::SessionState&,const strata::core::ModelGeometry&,void*){};
    std::vector<HipNativeChainStage> bindings={
        {{f.weights,f.model,front,{r0,r0,&cache,nullptr,2},pool,&pools0,123,8,4,0,zero},0},
        {{f.weights,f.model,f.session,{r1,r1,&cache,nullptr,2},pool,&pools1,123,8,4,0,zero},0}
    };
    {
        HipFinalVerifierExecutor stage(bindings);
        check(stage.prepare(f.plan,2,err),"grouped final native chain prepared");
        check(fake_cuda::allocations==3,"one mapped local handoff and one cross-runtime input/staging pair");
        check(stage.reset_request(1,err),"all stages reset");
        check(stage.begin_window({0,{5,6}},1,1,err),"grouped window admitted");
        Buffer input(2,f.plan.geometry); FinalStageResult output;
        const int copies=fake_cuda::copies;
        check(stage.run_final_window(f.frame(),input.view,output,err),"grouped head result");
        check(fake_cuda::copies==copies+1,"only cross-runtime input upload");
        check(fv::run_devices==std::vector<int>({0,1}) && pools0==1 && pools1==1,"native local run chain");
        check(fv::logit_devices==std::vector<int>({1,1}),"logits copied on head owner");
        check(stage.commit_prefix(1,1,1,err) && fv::commit_devices==std::vector<int>({0,1}),"native local prefix commit");
        check(stage.begin_window({1,{7,8}},1,2,err),"position advances by accepted prefix");
        check(stage.run_final_window(f.frame(2,1,2),input.view,output,err),"second grouped window");
        fv::fail_commit_device=1;
        check(!stage.commit_prefix(1,2,1,err),"later HIP commit failure is not ACKed");
        stage.abandon_window();
        stage.shutdown();
    }
    check(fake_cuda::allocations==0,"grouped graph and mapped handoff teardown");
    r0[4]=-1;
    {
        HipFinalVerifierExecutor stage(bindings);
        check(!stage.prepare(f.plan,2,err),"grouping refuses incomplete residency");
        stage.shutdown();
    }
    check(fake_cuda::allocations==0,"failed group allocation cleanup");
}
void fault_injection() {
    Fixture f;std::string err;
    for (int fault=0;fault<7;++fault) {
        fv::reset();fake_cuda::failure=0;fake_cuda::query=0;
        {
            auto b=f.bindings(); HipFinalVerifierExecutor stage(b);
            check(stage.prepare(f.plan,2,err),"fault fixture prepared");
            check(stage.begin_window({0,{1}},1,1,err),"fault fixture window admitted");
            Buffer in(1,f.plan.geometry); FinalStageResult result;
            if(fault==0)fake_cuda::failure=1;
            if(fault==1)fake_cuda::query=cudaErrorNotReady;
            if(fault==2)fv::run_fails=true;
            if(fault==3)fv::copy_fails=true;
            bool executed=stage.run_final_window(f.frame(1),in.view,result,err);
            if(fault<=3)check(!executed,"copy/run/fence/logits failure propagated");
            else {
                check(executed,"provisional result before commit fault");
                if(fault==4)fv::commit_fails=true;
                if(fault==5)fv::wait_fails=true;
                if(fault==6)fake_cuda::query=cudaErrorNotReady;
                check(!stage.commit_prefix(1,1,1,err),"commit or fence fault propagated");
            }
            fake_cuda::query=0;
            stage.abandon_window();stage.shutdown();
        }
        check(fake_cuda::allocations==0,"faulted stage allocations released");
    }
    fake_cuda::failure=0;fake_cuda::query=0;
    for (int fault=1;fault<=2;++fault) {
        fv::reset();
        auto b=f.bindings(); HipFinalVerifierExecutor stage(b);
        fake_cuda::fail_alloc=fake_cuda::alloc_calls+fault;
        check(!stage.prepare(f.plan,2,err),"partial CUDA allocation failure detected");
        fake_cuda::fail_alloc=0;
        stage.shutdown();
        check(fake_cuda::allocations==0,"partial allocation cleanup");
    }
    fv::reset();{
        fv::init_fails=true;
        auto b=f.bindings(); HipFinalVerifierExecutor stage(b);
        check(!stage.prepare(f.plan,2,err),"verifier init failure detected");
        stage.shutdown();
    }
    check(fake_cuda::allocations==0,"verifier init failure frees allocated staging");
}
}
int main() {
    try {
        success_and_replay(); invalid_inputs(); nonfinite_fail_closed(); fault_injection(); grouped_native_chain();
        std::cout << "HIP final verifier CPU-double tests: PASS (" << assertions << " assertions; no GPU)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "HIP final verifier CPU-double tests: FAIL " << e.what() << '\n';
        return 1;
    }
}
