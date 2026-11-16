#pragma once

// A front stage without a GPU, for the CPU tests: it records the calls and writes a mark into the hand-off memory.

#include "strata/hetero/front_link.hpp"

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace strata::hetero::test {

struct StubBackend : FrontBackend {
    std::vector<std::string> calls;
    LinkIdentity expect;
    bool check_identity = false;
    SharedRegion region;
    int commit_sleep_ms = 0;
    bool fail_next_run = false;

    bool attach(const LinkIdentity& id, SharedRegion&& r, std::string& err) override {
        if (check_identity && id != expect) { err = "identity mismatch"; return false; }
        region = std::move(r);
        calls.push_back("attach");
        return true;
    }
    bool run(int T, const int32_t* tokens, int64_t pos0, std::string& err) override {
        if (fail_next_run) { fail_next_run = false; err = "stub run failure"; return false; }
        calls.push_back("run " + std::to_string(T) + " " + std::to_string(pos0) + " " + std::to_string(tokens[T - 1]));
        // what a real front stage does from the GPU: the rows, then the ready word
        float* rows = static_cast<float*>(region.slot(0));
        for (int t = 0; t < T; ++t) rows[t] = (float) tokens[t] + 0.5f;
        uint32_t ready = (uint32_t) T;
        std::memcpy(region.slot(1), &ready, sizeof ready);
        return true;
    }
    bool commit(int n_keep, std::string&) override {
        if (commit_sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(commit_sleep_ms));
        calls.push_back("commit " + std::to_string(n_keep));
        return true;
    }
    bool prefill(const int32_t* tokens, int64_t n, int64_t pos0, bool last, std::string&) override {
        calls.push_back("prefill " + std::to_string(n) + " " + std::to_string(pos0) + " " + (last ? "last" : "more") + " " +
                        std::to_string(tokens[n - 1]));
        return true;
    }
    bool session(SessionOp op, int64_t a, int64_t b, std::string&) override {
        calls.push_back("session " + std::to_string((int) op) + " " + std::to_string(a) + " " + std::to_string(b));
        return true;
    }
};

inline LinkIdentity test_identity() {
    LinkIdentity id;
    id.model_fingerprint = 0x1234abcdULL;
    id.n_layers = 48;
    id.front_end = 15;
    id.hc = 4;
    id.n_embd = 2560;
    id.max_rows = 8;
    id.max_context = 262144;
    return id;
}

} // namespace strata::hetero::test
