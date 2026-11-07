#pragma once

// Experimental, backend-neutral final-stage result channel.
// This v1 codec is deliberately independent of the v4 activation/commit
// frames. It is NOT integrated with PipelineSession or GPU runtimes yet.
// Use only on a dedicated local AF_UNIX stream with trusted workers.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace strata::heterogeneous {

constexpr uint32_t final_result_max_rows = 8192;  // v4 WindowSpec limit
constexpr uint32_t final_result_max_payload_bytes = 16u * 1024u * 1024u;
constexpr size_t final_result_header_bytes = 64;

struct FinalStageResult {
    uint32_t stage_index = 0;
    uint64_t request_id = 0;
    uint64_t window_id = 0;
    uint64_t plan_fingerprint = 0;
    uint32_t rows = 0;
    uint32_t vocab_size = 0;
    // One model-predicted token per row, exactly as returned by the final
    // verifier's head. These are PROVISIONAL until a successful prefix commit.
    std::vector<int32_t> predicted_tokens;
    // Sorted, unique row indices for which the final head detected a NaN or
    // infinity. A nonempty set makes the result unsafe for acceptance.
    std::vector<uint32_t> nonfinite_rows;
    // If present, exactly rows*vocab_size float32 logits, row-major. Bitwise
    // transport; not a probability distribution or post-sampler guarantee.
    std::optional<std::vector<float>> logits;
};

// Throws std::invalid_argument for inconsistent shapes or values.
void validate_final_result(const FinalStageResult& result);
// Verifies a result belongs to the requested final stage/window/plan.
void require_final_result_identity(const FinalStageResult& result,
                                   uint32_t final_stage_index,
                                   uint64_t request_id, uint64_t window_id,
                                   uint64_t plan_fingerprint, uint32_t rows);
// Versioned little-endian wire packet: fixed 64-byte header, bounded payload,
// FNV-1a integrity over header and body excluding the 4-byte checksum field.
// Integrity protects against accidental corruption, not adversaries.
std::vector<uint8_t> encode_final_result(const FinalStageResult& result);
FinalStageResult decode_final_result(const std::vector<uint8_t>& bytes);

// Local AF_UNIX SOCK_STREAM helpers. Deadline is for the complete packet,
// not per read. A clean EOF before the header returns std::nullopt; truncated
// data, checksum mismatches and timeouts throw std::runtime_error.
void send_final_result(int fd, const FinalStageResult& result, int timeout_ms = 30000);
std::optional<FinalStageResult> receive_final_result(int fd, int timeout_ms = 30000);

// The result is never marked publishable until the coordinator has received
// matching commit ACKs from EVERY stage. This is a tiny helper for future
// opt-in final-stage integration; it does not perform commit or rollback.
class FinalResultGate {
public:
    void stage(FinalStageResult result, uint32_t final_stage_index,
               uint64_t request_id, uint64_t window_id,
               uint64_t plan_fingerprint, uint32_t rows);
    const FinalStageResult& provisional() const;
    void mark_committed(uint64_t request_id, uint64_t window_id);
    FinalStageResult take_committed();
    void poison() noexcept;
    bool poisoned() const noexcept { return poisoned_; }
    bool has_provisional() const noexcept { return staged_.has_value() && !committed_; }
private:
    std::optional<FinalStageResult> staged_;
    bool committed_ = false;
    bool poisoned_ = false;
};

} // namespace strata::heterogeneous
