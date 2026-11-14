#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>
namespace strata::kernels { constexpr int kVerifyMaxT = 8; }
namespace strata::core {
// Fail-closed M2.17 CPU-only test-double contract.
struct StageMicroProbe {
    uint32_t n_embd = 0, n_ff = 0, conv_channels = 0, conv_kernel = 0;
    std::vector<float> data;
    std::vector<uint8_t> input_q8, swiglu_q8;
};
struct StageBranchProbe {
    uint32_t n_embd = 0, n_expert = 0, k = 0;
    uint32_t conv_channels = 0, value_heads = 0, value_dim = 0;
    std::vector<float> data;
    std::vector<int32_t> ids;
};
// Fail-closed M2.12 diagnostic contract for CPU-only native test doubles.
struct LayerZeroArithmetic {
    uint32_t n_embd = 0, qkv_channels = 0;
    std::vector<float> embedded, mixed, qkv;
    std::vector<uint8_t> q8_bytes;
};
struct WeightTable {};
struct ModelGeometry { int64_t n_layers = 4, hc = 2, n_embd = 3, n_vocab = 100; };
struct SessionState { int64_t layer_lo = 0, layer_hi = 2, max_cells = 32; };
struct VerifyHits {};
using PoolMultiFn = void (*)(void*, const float*, const int32_t*, int64_t, int64_t, float*, int64_t);
namespace fake_verify {
inline int runs = 0, commits = 0, aborted = 0, keep = 0;
inline bool publish = false, initialized = false, fail_run = false, fail_commit = false;
inline const float* input = nullptr;
inline float* output = nullptr;
inline int rows = 0;
inline int64_t begin = 0, end = 0, pos = 0;
inline std::vector<int32_t> tokens;
}
class Verifier {
public:
    static constexpr int kStage0ProbeLayers = 12;
    static constexpr int kStage0ProbePhases = 4;
    void set_stage_micro_probe(bool) {}
    bool copy_stage_micro_probe(StageMicroProbe&, std::string& e) const {
        e = "stage microprobe unsupported in CPU test double"; return false;
    }
    void set_stage_branch_probe(bool) {}
    bool copy_stage_branch_probe(StageBranchProbe&, std::string& e) const {
        e = "branch probe unavailable in CPU test double"; return false;
    }
    void set_stage0_phase_probe(bool) {}
    bool copy_stage0_phase_probe(std::vector<float>&, std::string& e) const {
        e = "stage0 phase probe unsupported in CPU test double"; return false;
    }
    void set_layer_zero_arithmetic_probe(bool) {}
    bool copy_layer_zero_arithmetic(LayerZeroArithmetic&, std::string& e) const {
        e="layer-zero probe unsupported in CPU stub"; return false;
    }
    void set_taps(const std::vector<int>&) {}
    bool gather_taps_host(float*, int, int, std::string& err) {
        err = "tap collection is not implemented by this test double";
        return false;
    }
    void set_stage(int64_t b, int64_t e, const float* in, float* out) {
        fake_verify::begin = b; fake_verify::end = e;
        fake_verify::input = in; fake_verify::output = out;
    }
    void set_handoff_rows(int n) { fake_verify::rows = n; }
    void set_next(Verifier*, void*) {}
    void set_handoff_flags(uint32_t*, uint32_t*, uint32_t*, uint32_t*) {}
    void set_always_publish(bool b) { fake_verify::publish = b; }
    void set_head_sampling(bool) {}
    bool init(const WeightTable&, const ModelGeometry&, SessionState&, const VerifyHits&, const void*, int, std::string&) {
        fake_verify::initialized = true; return true;
    }
    void* stream() { return nullptr; }
    bool run(int n, const int32_t* t, int64_t pos, PoolMultiFn pool, void* user, int32_t*, std::string&) {
        ++fake_verify::runs; fake_verify::tokens.assign(t, t+n); fake_verify::pos = pos;
        pool(user, nullptr, nullptr, n, 0, nullptr, fake_verify::begin);
        for (int i = 0; i < n * 11; ++i)
            fake_verify::output[i] = fake_verify::input ? fake_verify::input[i] + 1 : 42;
        return !fake_verify::fail_run;
    }
    bool commit(int keep, std::string&) { ++fake_verify::commits; fake_verify::keep = keep; return !fake_verify::fail_commit; }
    bool release_gpu_waits(int) { ++fake_verify::aborted; return true; }
};
}
