#pragma once

// Host-only shared-page ABI for the experimental native ready-word edge.
// Keep the native verifier R/bo/inj layout: no new GPU synchronization protocol.
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::heterogeneous {
struct DirectHandoffLayout {
    static constexpr size_t alignment = 4096;
    static constexpr size_t max_payload_bytes = 16 * 1024 * 1024;
    size_t r_words, bo_words, inj_words, payload_bytes, ready_offset, error_offset, extent;

    static DirectHandoffLayout make(size_t hc, size_t n_embd, size_t rows) {
        // Check before multiplying; copy_from_mapped uses 16-byte aligned groups.
        constexpr size_t max_words = max_payload_bytes / sizeof(float);
        if (!hc || hc % 4 || !n_embd || n_embd % 4 || !rows || rows > 8 ||
            hc > max_words / rows || n_embd > max_words / rows)
            throw std::invalid_argument("invalid native handoff geometry");
        const size_t ht = hc * rows, nt = n_embd * rows;
        if (ht > max_words / n_embd) throw std::invalid_argument("native handoff exceeds cap");
        const size_t r = ht * n_embd;
        if (nt > max_words - r || ht > max_words - r - nt)
            throw std::invalid_argument("native handoff exceeds cap");
        const size_t bytes = (r + nt + ht) * sizeof(float);
        const size_t ready = (bytes + alignment - 1) / alignment * alignment;
        return {r, nt, ht, bytes, ready, ready + 64, ready + alignment};
    }
    size_t words() const noexcept { return r_words + bo_words + inj_words; }
};

// Unique, exactly representable FP32 test data for each round and word.
inline float direct_probe_value(uint32_t sequence, size_t word) noexcept {
    return static_cast<float>((uint64_t(sequence) * 131071u + word) & 0xffffffu);
}
} // namespace strata::heterogeneous
