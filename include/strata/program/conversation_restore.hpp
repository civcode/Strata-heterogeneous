#pragma once
// The conversation restore's admission / mutation gate (generate.cpp's serve loop).
//
// Why this exists: a restore writes into session buffers that already exist, but it still needs GPU copy
// resources (conversation_copy.hpp's private stream and ordering event per device). Acquiring them can fail -
// on AMD a cold stream is a ~146 MiB runtime allocation, measured on Radeon AI PRO R9700 - and `copy_stream_ready`
// is only a snapshot, not a reservation: another thread can take the pooled handle in between. The old guard
// predicted that allocation from a 64 MiB device-wide free-memory sample, which was both too small for a cold
// HIP stream and blind to the race, and it did not cover borrowed-prefix restores at all.
//
// So the resources are ACQUIRED, for every participating device, before anything is mutated - before the selected
// snapshot leaves the cache, before the outgoing conversation is parked, before checkpoint ownership moves and
// before the K/V grows. Acquisition puts the handles in the calling thread's own slots, where no other thread can
// take them, and they stay for the thread's lifetime (returned to the pool when it ends).
//
// Two failure classes, kept apart by `phase_`:
//   * before mutation (prepare refused): abandon the restore, keep the snapshot cached, read the prompt normally;
//   * after mutation (begin_mutation ran): the live state may already be partly overwritten, so the request is
//     aborted - never decoded from, never silently resumed.
//
// Preparation deliberately does NOT warm the ordering path (`order_after_queued`'s event record + legacy-stream
// wait): a legacy-stream operation can conflict with another thread's graph capture, and the record/wait would
// add dependencies for no benefit. So a later ordering or transfer failure is still possible and stays fatal.
#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_copy_resources.hpp"

#include <cstdio>
#include <optional>
#include <utility>

namespace strata::program {

class ConversationRestoreAttempt {
public:
    enum class Phase {
        idle,       // no restore: a normal request
        prepared,   // resources acquired, snapshot taken, mutation not started yet (internal only)
        declined,   // preparation refused: behaves exactly like idle, the snapshot stayed cached
        mutating,   // the live state is being overwritten: any failure aborts the request
        complete,   // restored
        failed      // aborted: nothing may run, the request must not continue
    };
    std::optional<core::SavedConversation> incoming;
    bool borrow = false;

    /// Validate nothing here: the caller validates the image first and passes the selection by `serial`.
    /// `prefix`: a borrowed-prefix restore (the donor stays parked). `acquire`: the real
    /// conversation_copy_prepare or a test double with the same signature. The snapshot leaves the cache only
    /// after acquisition succeeded, so a refusal needs no put-back and cannot lose it.
    template<class Prepare>
    core::ConversationCopyPreparation prepare(core::ConversationCache& cache, uint64_t serial, bool prefix,
                                             const int* devices, size_t count, Prepare&& acquire,
                                             std::string& error) {
        using Result = core::ConversationCopyPreparation;
        if (phase_ != Phase::idle) {
            error = "conversation restore: prepare called twice";
            phase_ = Phase::failed;
            return Result::fatal;
        }
        if (cache.find(serial) == nullptr) {   // cannot happen on the serving thread: loud, but still safe
            std::fprintf(stderr, "strata serve: conversation cache: the selected snapshot is gone before preparation\n");
            error = "conversation restore selection is no longer available";
            phase_ = Phase::declined;
            return Result::unavailable;
        }
        const Result result = acquire(devices, count, error);
        if (result != Result::ready) {
            // Nothing was taken and nothing was parked: the outgoing conversation and the cached snapshot are
            // both exactly as they were. unavailable -> read the prompt; fatal -> abort (a poisoned context is
            // not something a reread can recover from).
            phase_ = result == Result::unavailable ? Phase::declined : Phase::failed;
            return result;
        }
        const size_t index = cache.index_of(serial);
        if (index == SIZE_MAX) {
            std::fprintf(stderr, "strata serve: conversation cache: the selected snapshot vanished during preparation\n");
            error = "conversation restore selection changed during preparation";
            phase_ = Phase::declined;   // the acquired handles are this thread's now; nothing else changed
            return Result::unavailable;
        }
        borrow = prefix;
        if (!borrow) incoming.emplace(cache.take(index));
        phase_ = Phase::prepared;
        error.clear();
        return Result::ready;
    }

    /// Cross the boundary: from here a failure cannot be recovered by rereading the prompt.
    bool begin_mutation() {
        if (phase_ != Phase::prepared) { phase_ = Phase::failed; return false; }
        phase_ = Phase::mutating;
        return true;
    }

    /// A borrowed prefix becomes a whole restore when the outgoing conversation fits only in the donor's room.
    /// Legal only after the boundary, with the conservative device set already prepared; it takes the donor the
    /// same way `prepare` would have, so the conversion cannot bypass preparation.
    bool take_donor(core::ConversationCache& cache, uint64_t serial) {
        if (phase_ != Phase::mutating || !borrow) { phase_ = Phase::failed; return false; }
        const size_t index = cache.index_of(serial);
        if (index == SIZE_MAX) { phase_ = Phase::failed; return false; }
        incoming.emplace(cache.take(index));
        borrow = false;
        return true;
    }

    /// Run one step of the restore (or of an ordinary request). Refuses outright when the attempt already
    /// failed or has not crossed the boundary; otherwise runs it and records the failure. A declined attempt
    /// behaves like idle: the request keeps going as if no snapshot had matched.
    template<class Operation>
    bool apply(Operation&& operation) {
        if (phase_ == Phase::failed || phase_ == Phase::prepared) return false;
        if (operation()) return true;
        if (phase_ == Phase::mutating || phase_ == Phase::complete) phase_ = Phase::failed;
        return false;
    }

    /// Last gate before this request continues to the prompt path: false only for an attempt that broke.
    bool finish() {
        if (phase_ == Phase::mutating) phase_ = Phase::complete;
        return phase_ != Phase::failed && phase_ != Phase::prepared;
    }

    Phase phase() const { return phase_; }
    bool restoring() const { return phase_ == Phase::mutating || phase_ == Phase::complete; }
    bool abandoned() const { return phase_ == Phase::declined; }

private:
    Phase phase_ = Phase::idle;
};

/// The two coupled resume-source decisions the serve loop makes once the restore attempt's phase is known:
/// `pin_sibling` (pin=N: this request resumes exactly at the pinned shared prefix the session already holds, so
/// the outgoing conversation is NOT parked - N queries keep one prefix) and `park_outgoing`. Both were inline
/// expressions; they are pure and total here so the fallback can be checked rather than argued.
///
/// The property that matters for a DECLINED restore: its inputs are exactly the ones a request that never
/// matched a parked conversation would have (no incoming image, no borrow, `slot_source` left as the batch-slot
/// scan chose it). So a refusal falls back to the batch slot, or to the checkpoint chain when there is no slot -
/// both paths such a request already takes - and never to a half-applied restore.
struct ResumeGate { bool pin_sibling = false; bool park_outgoing = false; };
inline ResumeGate resume_gate(bool has_incoming, bool from_live, bool live_ok, int slot_source, int64_t resume,
                             int64_t req_pin, bool req_ckpt, int prompt_cache, bool park_worth,
                             bool pinned_at_resume) {
    ResumeGate g;
    g.pin_sibling = req_pin > 0 && req_ckpt && prompt_cache >= 3 && !from_live && !has_incoming &&
                    slot_source < 0 && resume == req_pin && live_ok && pinned_at_resume;
    g.park_outgoing = (!from_live || has_incoming || slot_source >= 0) && !g.pin_sibling && park_worth;
    return g;
}

}  // namespace strata::program
