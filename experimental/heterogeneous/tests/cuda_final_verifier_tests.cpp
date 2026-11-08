// Same final-CUDA adapter source compiled against CPU CUDA/Verifier doubles.
// No real runtime, weights, model loading or numerical GPU assertions.
#include "strata_heterogeneous/cuda_final_verifier.hpp"

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
        plan.stages = {{Backend::cuda, 0, 0, 2}, {Backend::cuda, 1, 2, 3}, {Backend::cuda, 2, 3, 4}};
    }
    CudaVerifierBindings bindings(int64_t initial=0, int timeout=4) {
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
        CudaFinalVerifierExecutor stage(bindings);
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
        auto b=f.bindings(); CudaFinalVerifierExecutor stage(b);
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
        if(test==0)plan.stages[2].backend=Backend::hip;
        if(test==1)plan.stages[2].end_layer=3;
        if(test==2)bindings.model_fingerprint=999;
        if(test==3)f.session.layer_lo=2;
        if(test==4)bindings.pool=nullptr;
        CudaFinalVerifierExecutor stage(bindings);
        check(!stage.prepare(plan,2,err),"bad model, backend or stage carve rejected");
        check(fake_cuda::allocations==0,"bad model rejected before allocation");
        f.session.layer_lo=3;
    }
}
void nonfinite_fail_closed() {
    fv::reset();Fixture f;std::string err;
    {
        auto b=f.bindings(); CudaFinalVerifierExecutor stage(b);
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
        auto b=f.bindings(); CudaFinalVerifierExecutor no_logits(b,{false});
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
void fault_injection() {
    Fixture f;std::string err;
    for (int fault=0;fault<7;++fault) {
        fv::reset();fake_cuda::failure=0;fake_cuda::query=0;
        {
            auto b=f.bindings(); CudaFinalVerifierExecutor stage(b);
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
        auto b=f.bindings(); CudaFinalVerifierExecutor stage(b);
        fake_cuda::fail_alloc=fake_cuda::alloc_calls+fault;
        check(!stage.prepare(f.plan,2,err),"partial CUDA allocation failure detected");
        fake_cuda::fail_alloc=0;
        stage.shutdown();
        check(fake_cuda::allocations==0,"partial allocation cleanup");
    }
    fv::reset();{
        fv::init_fails=true;
        auto b=f.bindings(); CudaFinalVerifierExecutor stage(b);
        check(!stage.prepare(f.plan,2,err),"verifier init failure detected");
        stage.shutdown();
    }
    check(fake_cuda::allocations==0,"verifier init failure frees allocated staging");
}
}
int main() {
    try {
        success_and_replay(); invalid_inputs(); nonfinite_fail_closed(); fault_injection();
        std::cout << "CUDA final verifier CPU-double tests: PASS (" << assertions << " assertions; no GPU)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CUDA final verifier CPU-double tests: FAIL " << e.what() << '\n';
        return 1;
    }
}
