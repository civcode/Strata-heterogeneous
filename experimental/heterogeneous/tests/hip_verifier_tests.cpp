#include "strata_heterogeneous/hip_verifier.hpp"
#include <iostream>
#include <stdexcept>
using namespace strata::heterogeneous;
namespace fv = strata::core::fake_verify;
#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
void pool(void* user, const float*, const int32_t*, int64_t, int64_t, float*, int64_t) {
    ++*static_cast<int*>(user);
}
int main() {
    try {
        strata::core::WeightTable weights;
        strata::core::ModelGeometry geometry;
        strata::core::SessionState session;
        int pool_calls = 0;
        const auto plan = parse_plan("hip:0:0-2,hip:1:2-3,hip:2:3-4", 4, {2,3}, 123);
        auto bindings = [&] { return HipVerifierBindings{weights, geometry, session, {}, pool, &pool_calls, 123, 8, 2}; };
        std::string error;
        // Real source compiled with CPU doubles. This is not a GPU parity test.
        for (size_t index : {size_t(0), size_t(1)}) {
            session.layer_lo = plan.stages[index].first_layer;
            session.layer_hi = plan.stages[index].end_layer;
            auto b = bindings(); b.initial_position = 4;
            HipVerifierDriver driver(b);
            DeviceStageExecutor executor(driver);
            REQUIRE(executor.prepare(plan, index, error));
            REQUIRE(fv::publish && fv::rows == 8 && fv::begin == session.layer_lo && fv::end == session.layer_hi);
            REQUIRE((fv::input != nullptr) == (index != 0));
            WindowSpec spec{4, {5,6}};
            REQUIRE(executor.begin_window(spec, 1, 1, error));
            spec.token_ids[0] = 99; // driver/executor must own their token copies
            Frame in;
            in.stage_index = index; in.request_id = 1; in.window_id = 1;
            in.plan_fingerprint = plan.fingerprint(); in.geometry = plan.geometry;
            in.rows = 2; in.payload.assign(22, 7);
            Frame out;
            const int copies_before = fake_cuda::copies;
            REQUIRE(executor.run_window(in, out, error));
            // Stage zero reads embeddings, not handoff_in. Only D2H is
            // needed there; middle stages still perform H2D + D2H.
            REQUIRE(fake_cuda::copies - copies_before == (index ? 2 : 1));
            REQUIRE(out.payload == std::vector<float>(22, index ? 8 : 42));
            REQUIRE(fv::tokens == std::vector<int32_t>({5,6}) && fv::pos == 4);
            REQUIRE(executor.commit_prefix(1, 1, 1, error));
            REQUIRE(fv::keep == 1);
            // T=1 must still defer commit (set_always_publish above).
            REQUIRE(executor.begin_window({5,{6}}, 1, 2, error));
            in.window_id = 2; in.rows = 1; in.payload.resize(11);
            Frame metadata = in; metadata.payload.clear();
            ActivationView view(reinterpret_cast<uint8_t*>(in.payload.data()),
                                in.payload.size() * sizeof(float), in.geometry, in.rows);
            REQUIRE(executor.run_staged_window(metadata, view, error));
            REQUIRE(in.payload == std::vector<float>(11, index ? 8 : 42));
            const int commits = fv::commits;
            REQUIRE(!executor.commit_prefix(1, 2, 0, error));
            REQUIRE(fv::commits == commits && executor.poisoned());
        }
        REQUIRE(pool_calls == 4 && fake_cuda::allocations == 0);
        session.layer_lo = 0; session.layer_hi = 2;
        for (int fault = 0; fault < 13; ++fault) {
            // Upload faults must use a middle stage: layer zero has no H2D.
            const size_t index = fault == 3 ? 1 : 0;
            session.layer_lo = plan.stages[index].first_layer;
            session.layer_hi = plan.stages[index].end_layer;
            HipVerifierDriver driver(bindings()); DeviceStageExecutor executor(driver);
            REQUIRE(executor.prepare(plan, index, error));
            WindowSpec spec{0,{1}};
            if (fault == 0) spec.token_ids = {100}; // vocab
            if (fault == 1) spec.pos0 = 32; // context
            if (fault == 2) spec.token_ids.assign(9,1); // max_t
            if (fault == 8) spec.pos0 = 1; // missing state at skipped position
            if (fault < 3 || fault == 8) { REQUIRE(!executor.begin_window(spec,1,1,error)); continue; }
            if (fault != 9) REQUIRE(executor.begin_window(spec,1,1,error));
            Frame in; in.stage_index = index;
            in.geometry = plan.geometry; in.rows = 1; in.payload.assign(11,0);
            in.request_id = 1; in.window_id = 1; in.plan_fingerprint = plan.fingerprint();
            if (fault == 3) fake_cuda::failure = 1;
            if (fault == 4) fv::fail_run = true;
            if (fault == 5) fake_cuda::query = cudaErrorNotReady; // bounded fence timeout
            if (fault == 6) fake_cuda::fail_download = true;
            Frame out;
            if (fault >= 10) {
                REQUIRE(executor.run_window(in,out,error));
                REQUIRE(executor.commit_prefix(1,1,1,error));
                WindowSpec next{fault == 10 ? 0 : 1, {2}};
                REQUIRE(!executor.begin_window(next, fault == 11 ? 2 : 1,
                                                fault == 12 ? 1 : 2, error));
            } else if (fault == 7) {
                REQUIRE(executor.run_window(in,out,error));
                fv::fail_commit = true;
                REQUIRE(!executor.commit_prefix(1,1,1,error));
            } else {
                REQUIRE(!executor.run_window(in,out,error));
                REQUIRE(out.payload.empty());
            }
            REQUIRE(executor.poisoned());
            fake_cuda::failure = 0; fv::fail_run = false; fake_cuda::query = 0;
            fake_cuda::fail_download = false; fv::fail_commit = false;
        }
        REQUIRE(fake_cuda::allocations == 0);
        for (size_t index : {size_t(0), size_t(1)}) {
            session.layer_lo = plan.stages[index].first_layer;
            session.layer_hi = plan.stages[index].end_layer;
            const int allocations = index ? 3 : 2;
            for (int allocation = 1; allocation <= allocations; ++allocation) {
                HipVerifierDriver driver(bindings()); DeviceStageExecutor executor(driver);
                fake_cuda::fail_alloc = fake_cuda::alloc_calls + allocation;
                REQUIRE(!executor.prepare(plan,index,error));
                executor.shutdown();
                REQUIRE(fake_cuda::allocations == 0);
                fake_cuda::fail_alloc = 0;
            }
        }
        session.layer_lo = 0; session.layer_hi = 2;
        // Reject final stage, model mismatch, wrong carve, and CUDA before allocations.
        for (int fault = 0; fault < 4; ++fault) {
            auto b = bindings(); if (fault == 1) b.model_fingerprint = 456;
            auto p = plan; if (fault == 3) p.stages[0].backend = Backend::cuda;
            if (fault == 2) session.layer_hi = 3;
            HipVerifierDriver driver(b); DeviceStageExecutor executor(driver);
            REQUIRE(!executor.prepare(p, fault == 0 ? 2 : 0, error));
            REQUIRE(fake_cuda::allocations == 0);
            session.layer_hi = 2;
        }
        geometry.hc = 1000000000; geometry.n_embd = 1000000000;
        {
            auto huge = plan; huge.geometry = {1000000000,1000000000};
            HipVerifierDriver driver(bindings()); DeviceStageExecutor executor(driver);
            REQUIRE(!executor.prepare(huge,0,error));
            REQUIRE(fake_cuda::allocations == 0);
        }
        REQUIRE(fake_cuda::allocations == 0);
        std::cout << "PASS: HIP adapter lifecycle with CPU doubles (no GPU inference)\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
