#pragma once

// Shared state and pure decisions of the native --pipeline-windows 2 loop.
// Imported from the existing generate.cpp scheduler without changing its
// execution, snapshot/rollback, stream operations or output publication.
#include <algorithm>
#include <cstdint>

namespace strata::core {
struct PipelineWindow {
    int T = 0, seq = 0;
    int64_t p = 0;
    int32_t tok[8] = {};
    float prob[8] = {};    // prob[i]: the drafter's probability of tok[i + 1]
    float p_on = 0.0f;     // previous window accepted whole, tok[0] its bonus
    bool ready = false, launched = false, finished = false, s1 = false, s1_done = false;
    int fr = 0;            // front stages done
    bool fl = false;       // in flight on front stage fr
    uint32_t com = 0;      // speculative front-stage commits, bit per stage
    bool spec = false;
    bool made = false;
    bool sfx = false;
    int sfx_match = 0;
};

// Native caller guarantees 1 <= spec <= 8 and valid chain arrays/ranges.
inline int pipeline_window_size(const float* pr, int from, int avail, int spec, double min_p) {
    int T = spec;
    if (min_p > 0.0) {
        T = 1;
        while (T < spec && from + T - 1 < avail && pr[from + T - 1] >= (float) min_p) ++T;
    }
    return std::max(1, std::min(T, 1 + avail - from));
}

inline PipelineWindow pipeline_make_lookahead(const PipelineWindow& A, const int32_t* oc,
                                               const float* op, int base, int avail, float pa,
                                               int spec, double min_p) {
    PipelineWindow B;
    if (base >= avail) return B;
    B.seq = A.seq + 1;
    B.p = A.p + A.T;
    B.tok[0] = oc[base];
    B.T = pipeline_window_size(op, base + 1, avail, spec, min_p);
    for (int i = 1; i < B.T; ++i) {
        B.tok[i] = oc[base + i];
        B.prob[i - 1] = op[base + i];
    }
    B.p_on = pa * op[base];
    B.ready = true;
    B.made = true;
    return B;
}

// Number of accepted draft tokens (the first verified token is separate).
// The native caller still applies its stop bound and nonfinite verdict.
inline int pipeline_accepted_drafts(const PipelineWindow& A, const int32_t* out) {
    int a = 0;
    while (a < A.T - 1 && A.tok[a + 1] == out[a]) ++a;
    return a;
}

// Reuse the native full-window/bonus-token condition. The caller MUST also
// check that B actually launched, the verdict is finite, and rollback/commit
// rules were satisfied; matching tokens alone never authorize publication.
inline bool pipeline_matches_lookahead(const PipelineWindow& A, const PipelineWindow& B,
                                       int a, const int32_t* out) {
    return a == A.T - 1 && B.tok[0] == out[A.T - 1];
}
} // namespace strata::core
