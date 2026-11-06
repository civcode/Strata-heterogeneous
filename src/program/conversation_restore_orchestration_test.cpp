// src/program/conversation_restore_orchestration_test.cpp - the conversation restore's ORCHESTRATION, not just
// its gate: the order generate.cpp's serve loop runs in, against the real ConversationCache, the real
// ConversationRestoreAttempt and the real resume_gate, with injected preparation and per-step failures.
//
// The serve loop is one very large function around a live model, so it cannot be linked into a test. This
// harness transcribes its call order and asserts the observable state at every step; the generate.cpp line each
// step mirrors is named in its comment. CPU-only: no GPU, no model, no VRAM, nothing exhausted.
//
// What it proves:
//   * preparation runs before ANY mutation (cache take, outgoing park, slot copy, K/V growth, session write);
//   * a refused preparation leaves the snapshot cached, the donor parked and the outgoing conversation
//     untouched, and serves the request exactly as one that never matched a parked conversation would be
//     served - the batch slot or the checkpoint chain, whichever that request would have used;
//   * after the mutation boundary a failed step aborts the request: nothing further runs, no decode follows,
//     and a half-written session is never marked valid;
//   * a later GPU's refusal cannot leave an apparently successful partial restore.
#include "strata/program/conversation_restore.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using strata::core::ConversationCache;
using strata::core::ConversationCopyPreparation;
using strata::core::ConversationImageKey;
using strata::core::SavedConversation;
using strata::program::ConversationRestoreAttempt;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::printf("  FAIL %s\n", what); ++g_fail; }
}

// ---- the state one request acts on ---------------------------------------------------------------
struct World {
    ConversationCache cache{64ull << 20, 4};
    std::vector<uint8_t> kv;                 // the live session's K/V: what a partial restore would corrupt
    std::vector<int32_t> live_ids{1, 2};     // the outgoing conversation
    int64_t resume = 0;
    bool from_live = false;
    bool live_ok = true;
    bool outgoing_parked = false;            // park_current ran
    size_t parked_held = 0;
    int slot_source = -1;                    // the batch slot the scan chose as a resume source
    int64_t slot_tokens = 0;
    bool slot_copied = false;
    bool kv_grown = false;                   // kvg_ensure
    bool decoded = false;                    // the request reached generation
    bool guarded = false;                    // the borrow's donor guard is held right now
    bool checkpoint_mounted = false;         // the chain gave the resume point
    std::vector<std::string> log;            // the order the steps happened in
    std::vector<std::thread::id> threads;    // every step's thread
    std::string err;                         // what the caller would have been told
    ConversationRestoreAttempt::Phase phase = ConversationRestoreAttempt::Phase::idle;
    bool abandoned = false;
    int prepare_calls = 0;
};

struct Options {
    std::vector<int32_t> prompt;
    int stages = 0;                          // later-stage GPUs, ids 1..stages
    ConversationCopyPreparation prepare = ConversationCopyPreparation::ready;
    int fail_prepare_at = -1;                // the device index whose acquisition fails
    const char* fail_step = nullptr;         // "park" "kvg" "restore" "stage" "stage_sync" "borrow"
    bool park_blocked = false;               // the donor's room is needed: a borrow converts to a whole restore
    bool invalid_image = false;
    int slot_source = -1;                    // the batch-slot scan's result
    int64_t req_pin = 0;                     // pin=N
    bool req_ckpt = true;
    int prompt_cache = 3;
    bool park_worth = true;
};

// 8 bytes each: the session's own K/V before any restore, and the parked snapshot's
const std::vector<uint8_t> kOutgoing(8, 0x11);
const std::vector<uint8_t> kPayload(8, 0xA5);

// live {1,2,3,4,5} with checkpoints at {1,2,3} and {1,2,3,4}: a prompt of {1,2,3,9} matches the shorter
// checkpoint while a deeper one exists, which is what ConversationCache::borrows asks for; {1,2,3,4,5,6}
// matches the live state and is taken whole.
void put_fixture(World& w) {
    SavedConversation img;
    img.live.ids = {1, 2, 3, 4, 5};
    img.live.gdn = kPayload;
    strata::core::ConversationCheckpoint shallow, deep;
    shallow.ids = {1, 2, 3};
    shallow.gdn = kPayload;
    deep.ids = {1, 2, 3, 4};
    deep.gdn = kPayload;
    img.checkpoints.push_back(shallow);
    img.checkpoints.push_back(deep);
    w.cache.put(std::move(img));
}

// the serve loop's restore half, in its order
bool run_request(World& w, const Options& o) {
    const std::vector<ConversationImageKey> no_imgs;
    w.kv = kOutgoing;
    w.slot_source = o.slot_source;
    w.slot_tokens = o.slot_source >= 0 ? 4 : 0;
    w.threads.push_back(std::this_thread::get_id());

    // (15970) the cache's best match, and whether it is a borrow
    const auto parked = w.cache.best(o.prompt, no_imgs, true);
    ConversationRestoreAttempt attempt;
    auto& incoming = attempt.incoming;
    bool& borrow = attempt.borrow;
    const bool selected_prefix = o.stages == 0 && parked.tokens > std::max(w.resume, w.slot_tokens) &&
                                 w.cache.borrows(parked);
    const bool wants_snapshot = parked.tokens > std::max(w.resume, w.slot_tokens);

    // (15978) validate while the image is still cached: an invalid one is discarded and never prepared
    const SavedConversation* selected = wants_snapshot ? w.cache.find(parked.serial) : nullptr;
    if (selected && o.invalid_image) { selected = nullptr; w.log.emplace_back("discard-invalid"); }
    if (wants_snapshot && !selected && !selected_prefix) {
        const size_t at = w.cache.index_of(parked.serial);
        if (at != SIZE_MAX) { (void) w.cache.take(at); w.log.emplace_back("evict-invalid"); }
    }

    if (selected) {
        std::vector<int> devices{-1};
        for (int s = 1; s <= o.stages; ++s) devices.push_back(s);
        std::string error;
        // (16013) ACTUAL resource acquisition: before take(), before parking, before the K/V grows
        auto acquire = [&](const int*, size_t count, std::string& err) {
            ++w.prepare_calls;
            w.threads.push_back(std::this_thread::get_id());
            w.log.emplace_back("prepare");
            check(!w.outgoing_parked, "prepare sees the outgoing conversation unparked");
            check(!w.slot_copied, "prepare sees no batch-slot copy");
            check(!w.kv_grown, "prepare sees the K/V ungrown");
            check(w.kv == kOutgoing, "prepare sees the session K/V untouched");
            if (o.fail_prepare_at >= 0 && (size_t) o.fail_prepare_at < count) {
                err = "injected acquisition failure";
                return o.prepare;
            }
            err.clear();
            return ConversationCopyPreparation::ready;
        };
        const auto prepared = attempt.prepare(w.cache, parked.serial, selected_prefix, devices.data(),
                                              devices.size(), acquire, error);
        if (prepared == ConversationCopyPreparation::fatal) { w.err = error; return false; }
        if (prepared == ConversationCopyPreparation::unavailable) { w.err = error; w.log.emplace_back("declined"); }
    }
    check(w.prepare_calls == (selected ? 1 : 0), "preparation runs exactly once per selection");

    // (16029) the mutation boundary
    if (incoming || borrow) {
        w.slot_source = -1;
        w.log.emplace_back("drop-slot");
        if (!attempt.begin_mutation()) { w.err = "not prepared"; return false; }
        w.log.emplace_back("boundary");
    }

    // (16043) pin=N and (16055) the park decision, as the one pure call generate.cpp makes
    bool pinned_at_resume = false;
    if (o.req_pin > 0 && w.resume == o.req_pin) pinned_at_resume = true;
    const auto rsel = strata::program::resume_gate(incoming.has_value(), w.from_live, w.live_ok, w.slot_source,
                                                   w.resume, o.req_pin, o.req_ckpt, o.prompt_cache, o.park_worth,
                                                   pinned_at_resume);
    auto fails = [&](const char* step) { return o.fail_step != nullptr && step == std::string(o.fail_step); };
    auto park = [&](size_t held) {
        w.threads.push_back(std::this_thread::get_id());
        w.log.emplace_back("park");
        if (fails("park")) return false;
        w.outgoing_parked = true;
        w.parked_held = held;
        return true;
    };

    // (16070) a borrow parks the outgoing conversation with its donor guarded
    if (borrow && !rsel.pin_sibling && o.park_worth) {
        w.guarded = true;
        const bool parked_ok = attempt.apply([&] { return park(0); });
        w.guarded = false;
        if (!parked_ok) { w.err = "park"; return false; }
        if (o.park_blocked) {
            // (16084) the outgoing state fits only in the donor's room: the borrow becomes a whole restore
            w.log.emplace_back("convert");
            if (!attempt.take_donor(w.cache, parked.serial)) { w.err = "donor gone"; return false; }
            if (!attempt.apply([&] { return park(incoming->bytes()); })) { w.err = "park after conversion"; return false; }
        }
    } else if (rsel.park_outgoing && !attempt.apply([&] { return park(incoming ? incoming->bytes() : 0); })) {
        w.err = "park";
        return false;
    }

    // (16098) the batch slot as the resume source - what a declined restore falls back to
    if (w.slot_source >= 0) {
        w.log.emplace_back("slot-copy");
        w.slot_copied = true;
        w.resume = w.slot_tokens;
        w.from_live = true;
    } else if (w.resume > 0) {
        w.log.emplace_back("mount-checkpoint");
        w.checkpoint_mounted = true;
    }

    // (16189) the elastic K/V grows before the snapshot is written in
    if (incoming && !attempt.apply([&] {
            w.log.emplace_back("kvg");
            if (fails("kvg")) return false;
            w.kv_grown = true;
            return true;
        })) { w.err = "kvg"; return false; }

    // (16200) a borrowed prefix, restored from the donor, which stays parked
    if (borrow && !attempt.apply([&] {
            w.log.emplace_back("borrow-restore");
            w.threads.push_back(std::this_thread::get_id());
            if (fails("borrow")) return false;
            const SavedConversation* donor = w.cache.find(parked.serial);
            if (donor == nullptr || donor->checkpoints.empty()) return false;
            w.kv = donor->checkpoints.front().gdn;
            w.resume = parked.tokens;
            w.checkpoint_mounted = true;
            return true;
        })) { w.err = "borrowed restore"; return false; }

    // (16228) the whole snapshot, then (16240) each later stage, then (16249) its synchronization
    if (incoming) {
        if (!attempt.apply([&] {
                w.log.emplace_back("restore");
                w.threads.push_back(std::this_thread::get_id());
                const auto& payload = incoming->live.gdn;
                w.kv.assign(payload.size(), 0x11);
                for (size_t i = 0; i < payload.size(); ++i) {
                    if (fails("restore") && i == payload.size() / 2) return false;   // stop half written
                    w.kv[i] = payload[i];
                }
                w.resume = (int64_t) incoming->live.ids.size();
                w.live_ids = incoming->live.ids;
                w.from_live = true;
                return true;
            })) { w.err = "restore"; return false; }
        for (int s = 1; s <= o.stages; ++s) {
            if (!attempt.apply([&] {
                    w.log.emplace_back("stage-restore");
                    w.threads.push_back(std::this_thread::get_id());
                    return !fails("stage");
                })) { w.err = "stage restore"; return false; }
            if (!attempt.apply([&] {
                    w.log.emplace_back("stage-sync");
                    return !fails("stage_sync");
                })) { w.err = "stage sync"; return false; }
        }
    }

    // (16407) the last gate before this request goes on to read its prompt and decode
    if (!attempt.finish()) { w.err = "incomplete restore"; return false; }
    w.log.emplace_back("decode");
    w.decoded = true;
    w.phase = attempt.phase();
    w.abandoned = attempt.abandoned();
    return true;
}

bool has(const World& w, const char* step) {
    for (const auto& s : w.log) if (s == step) return true;
    return false;
}
bool only(const World& w, const std::vector<const char*>& allowed) {
    for (const auto& s : w.log) {
        bool ok = false;
        for (const char* a : allowed) if (s == a) ok = true;
        if (!ok) return false;
    }
    return true;
}
}  // namespace

int main() {
    const std::vector<int32_t> whole{1, 2, 3, 4, 5, 6}, prefix{1, 2, 3, 9};

    {   // 1. a whole restore, prepared and taken before anything is mutated
        World w; put_fixture(w);
        Options o; o.prompt = whole; o.stages = 1;
        check(run_request(w, o), "whole restore succeeds");
        check(w.cache.size() == 0, "the snapshot left the cache");
        check(w.outgoing_parked && w.parked_held > 0, "the outgoing conversation parked, sized by the snapshot");
        check(w.kv_grown && w.kv == kPayload && w.resume == 5, "the session holds the restored bytes");
        check(w.decoded && w.phase == ConversationRestoreAttempt::Phase::complete, "restored, then decoded");
        const std::vector<std::string> want{"prepare", "drop-slot", "boundary", "park", "kvg", "restore",
                                            "stage-restore", "stage-sync", "decode"};
        check(w.log == want, "the steps ran in the serve loop's order");
        for (const auto& t : w.threads) check(t == w.threads.front(), "preparation and restore share one thread");
    }
    {   // 2. a borrowed prefix: the donor stays parked with its own checkpoints
        World w; put_fixture(w);
        Options o; o.prompt = prefix;
        check(run_request(w, o), "borrowed-prefix restore succeeds");
        check(w.cache.size() == 1 && w.cache.find(1) != nullptr, "the donor is still parked and findable");
        check(w.kv == kPayload && w.resume == 3 && w.checkpoint_mounted, "the prefix landed");
        check(w.outgoing_parked && w.parked_held == 0, "the borrow parks without reserving the donor's room");
        check(!w.guarded && w.decoded && !has(w, "convert"), "the guard was released, no conversion, decoded");
    }
    {   // 3. a blocked borrow converts to a whole restore without bypassing preparation
        World w; put_fixture(w);
        Options o; o.prompt = prefix; o.park_blocked = true;
        check(run_request(w, o), "the conversion succeeds");
        check(w.cache.size() == 0 && w.kv == kPayload && w.resume == 5, "the donor was taken whole and restored");
        check(has(w, "convert") && has(w, "prepare"), "it converted");
        size_t at_prepare = 0, at_convert = 0;
        for (size_t i = 0; i < w.log.size(); ++i) {
            if (w.log[i] == "prepare") at_prepare = i;
            if (w.log[i] == "convert") at_convert = i;
        }
        check(at_prepare < at_convert, "preparation preceded the conversion");
    }
    {   // 4. preparation refused on the FIRST device: nothing at all happened
        World w; put_fixture(w);
        Options o; o.prompt = whole; o.stages = 1; o.fail_prepare_at = 0;
        o.prepare = ConversationCopyPreparation::unavailable;
        check(run_request(w, o), "a refused preparation still serves the request");
        check(w.cache.size() == 1 && w.cache.find(1) != nullptr, "the snapshot is still cached, not lost");
        // the outgoing conversation still parks - that is the ordinary request's own decision, not a restore
        // mutation, and scenario 6 shows it is byte-for-byte what a request with no parked match does.
        check(!w.slot_copied && !w.kv_grown, "no restore mutation ran");
        check(!has(w, "drop-slot") && !has(w, "boundary") && !has(w, "restore"), "the restore never started");
        check(w.kv == kOutgoing && w.resume == 0 && w.live_ids == (std::vector<int32_t>{1, 2}),
              "the outgoing conversation and its K/V are intact");
        check(w.decoded && w.abandoned, "the prompt is read instead");
        check(only(w, {"prepare", "declined", "park", "mount-checkpoint", "decode"}), "no mutation step ran");
    }
    {   // 5. a LATER device's refusal: no partial restore; and fatal aborts rather than falling back
        World w; put_fixture(w);
        Options o; o.prompt = whole; o.stages = 1; o.fail_prepare_at = 1;
        o.prepare = ConversationCopyPreparation::unavailable;
        check(run_request(w, o), "a later GPU's refusal still serves the request");
        check(w.cache.size() == 1 && !w.kv_grown && w.kv == kOutgoing && !has(w, "restore"),
              "a later GPU's refusal leaves no partial restore and no restore mutation");
        World w2; put_fixture(w2);
        Options o2 = o; o2.prepare = ConversationCopyPreparation::fatal;
        check(!run_request(w2, o2), "a fatal preparation aborts the request");
        check(!w2.decoded && !w2.outgoing_parked && w2.cache.size() == 1 && w2.kv == kOutgoing,
              "a fatal preparation mutates nothing and never decodes");
        check(!w2.err.empty(), "the fatal reason reaches the caller");
    }
    {   // 6. the declined fallback IS the no-match request: same slot, same pin=N, same outcome
        for (const int slot : {-1, 2}) for (const int64_t pin : {(int64_t) 0, (int64_t) 2}) {
            World a; put_fixture(a);
            a.resume = pin;                       // the reused prefix the request already has
            Options o; o.prompt = whole; o.slot_source = slot; o.req_pin = pin;
            o.fail_prepare_at = 0; o.prepare = ConversationCopyPreparation::unavailable;
            check(run_request(a, o), "the declined request is served");

            World b; put_fixture(b);
            b.resume = pin;
            Options p = o; p.prompt = {7, 8, 9};  // matches nothing: the reference request
            p.fail_prepare_at = -1; p.prepare = ConversationCopyPreparation::ready;
            check(run_request(b, p), "the reference request is served");
            check(b.prepare_calls == 0, "the reference request prepares nothing");

            check(a.slot_copied == b.slot_copied && a.outgoing_parked == b.outgoing_parked &&
                      a.checkpoint_mounted == b.checkpoint_mounted && a.resume == b.resume &&
                      a.kv == b.kv && a.decoded == b.decoded,
                  "a declined restore falls back exactly like a request with no parked match");
            if (slot >= 0) check(a.slot_copied && a.resume == a.slot_tokens, "the batch slot is the resume source");
            if (slot < 0 && pin > 0) check(!a.outgoing_parked, "pin=N keeps the outgoing conversation unparked");
            if (slot < 0 && pin == 0) check(a.outgoing_parked, "without pin=N the outgoing conversation parks");
        }
    }
    {   // 7. a post-mutation restore failure: a half-written session is never decoded or marked valid
        World w; put_fixture(w);
        Options o; o.prompt = whole; o.stages = 1; o.fail_step = "restore";
        check(!run_request(w, o), "the failed restore aborts the request");
        check(!w.decoded, "no generation follows a partial restore");
        check(w.kv != kPayload && w.kv != kOutgoing, "the session really was partly overwritten");
        check(!w.from_live && w.resume == 0, "the partially restored session is not marked valid");
        check(!has(w, "stage-restore") && !has(w, "stage-sync"), "no later step ran after the failure");
    }
    for (const char* step : {"park", "kvg", "stage", "stage_sync"}) {
        World w; put_fixture(w);
        Options o; o.prompt = whole; o.stages = 1; o.fail_step = step;
        check(!run_request(w, o), "a failure after the boundary aborts the request");
        check(!w.decoded && w.phase == ConversationRestoreAttempt::Phase::idle,
              "an aborted request never reaches the decode gate");
    }
    {   // 8. an invalid image is discarded before preparation and prepares nothing
        World w; put_fixture(w);
        Options o; o.prompt = whole; o.invalid_image = true;
        check(run_request(w, o), "an invalid image still serves the request");
        check(w.cache.size() == 0 && w.decoded, "it was evicted and the prompt is read");
        check(w.prepare_calls == 0 && !has(w, "prepare"), "an invalid image never reaches preparation");
    }
    {   // 9. a borrow's donor survives a failure of the borrowed restore itself
        World w; put_fixture(w);
        Options o; o.prompt = prefix; o.fail_step = "borrow";
        check(!run_request(w, o), "a failed borrowed restore aborts");
        check(!w.decoded && w.cache.size() == 1 && w.cache.find(1) != nullptr, "the donor is still parked and whole");
        check(w.kv != kPayload, "the donor's bytes were not applied");
    }
    std::printf(g_fail ? "conversation_restore_orchestration_test: FAIL (%d)\n"
                       : "conversation_restore_orchestration_test: PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
