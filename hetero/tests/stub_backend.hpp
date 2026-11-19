#pragma once

// A front stage without a GPU, for the CPU tests: it records the calls and writes a mark into the hand-off memory.

#include "strata/hetero/front_link.hpp"
#include "strata/hetero/front_session.hpp"

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
    FrontImages last_images;
    bool images(FrontImages& im, std::string&) override {
        calls.push_back("images " + std::to_string(im.n) + " " + std::to_string(im.images.size()) + " " + std::to_string(im.rows.size()));
        last_images = std::move(im);
        return true;
    }
    /// pipelined windows: parity q writes its own hand-off (slot 3q, or slot 0 in a two-slot region) and ready word
    bool pl_run(int q, int T, const int32_t* tokens, int64_t pos0, std::string& err) override {
        if (fail_next_run) { fail_next_run = false; err = "stub run failure"; return false; }
        calls.push_back("pl_run " + std::to_string(q) + " " + std::to_string(T) + " " + std::to_string(pos0) + " " +
                        std::to_string(tokens[T - 1]));
        const size_t hs = region.slots() >= 5 && q == 1 ? 3 : 0;
        float* rows = static_cast<float*>(region.slot(hs));
        for (int t = 0; t < T; ++t) rows[t] = (float) tokens[t] + 0.25f;
        uint32_t ready = (uint32_t) (100 + T);
        std::memcpy(region.slot(hs + 1), &ready, sizeof ready);
        return true;
    }
    bool pl_commit(int q, int n_keep, std::string&) override {
        calls.push_back("pl_commit " + std::to_string(q) + " " + std::to_string(n_keep));
        return true;
    }
    bool pl_undo(int q, std::string&) override {
        calls.push_back("pl_undo " + std::to_string(q));
        return true;
    }
    /// a prompt chunk read ahead: buffer b is slot 2 (b = 0) or slot 5 (b = 1, a six-slot region); its first float is the
    /// chunk's last token + 0.75.  `fail_next_pf`: the read fails, which only the wait reports (as a real worker's job)
    bool fail_next_pf = false;
    std::string pf_err;
    bool prefill_start(const int32_t* tokens, int64_t n, int64_t pos0, int buf, std::string&) override {
        calls.push_back("pf_start " + std::to_string(n) + " " + std::to_string(pos0) + " " + std::to_string(buf) + " " +
                        std::to_string(tokens[n - 1]));
        if (fail_next_pf) { fail_next_pf = false; pf_err = "stub prompt failure"; return true; }
        const size_t slot = buf == 1 && region.slots() >= 6 ? 5 : 2;
        static_cast<float*>(region.slot(slot))[0] = (float) tokens[n - 1] + 0.75f;
        return true;
    }
    bool prefill_wait(int buf, std::string& err) override {
        calls.push_back("pf_wait " + std::to_string(buf));
        if (!pf_err.empty()) { err = pf_err; pf_err.clear(); return false; }
        return true;
    }
};

/// The front stage's session bookkeeping as the engine's front backend uses it, with a string for the session state:
/// a checkpoint is the token history (and pictures) it was taken at, an image that history with its checkpoints.
struct BookBackend : StubBackend {
    struct Part { std::vector<int32_t> ids; std::vector<ImageKey> imgs; };
    struct Image { std::vector<int32_t> ids; std::vector<ImageKey> imgs; std::vector<Part> parts; };
    FrontSessionBook<Part, Image> book{3, 2};
    std::string state;   // what the "session" holds: the tokens, as text
    static std::string text(const std::vector<int32_t>& ids) {
        std::string t;
        for (int32_t v : ids) t += std::to_string(v) + ",";
        return t;
    }
    int64_t key_with(const Part& p, bool steer) const { return state_key(p.ids.data(), (int64_t) p.ids.size(), steer, p.imgs); }
    bool run(int T, const int32_t* tokens, int64_t pos0, std::string& err) override {
        return book.on_run(tokens, T, pos0, err) && StubBackend::run(T, tokens, pos0, err);
    }
    bool commit(int n_keep, std::string& err) override {
        book.on_commit(n_keep);
        state = text(book.history());
        return StubBackend::commit(n_keep, err);
    }
    bool prefill(const int32_t* tokens, int64_t n, int64_t pos0, bool last, std::string& err) override {
        if (!book.on_prefill(tokens, n, pos0, err)) return false;
        state = text(book.history());
        return StubBackend::prefill(tokens, n, pos0, last, err);
    }
    bool prefill_start(const int32_t* tokens, int64_t n, int64_t pos0, int buf, std::string& err) override {
        if (!book.on_prefill(tokens, n, pos0, err)) return false;
        return StubBackend::prefill_start(tokens, n, pos0, buf, err);
    }
    bool pl_run(int q, int T, const int32_t* tokens, int64_t pos0, std::string& err) override {
        return book.on_pl_run(q, tokens, T, pos0, err) && StubBackend::pl_run(q, T, tokens, pos0, err);
    }
    bool pl_commit(int q, int n_keep, std::string& err) override {
        if (!book.on_pl_commit(q, n_keep, err)) return false;
        state = text(book.history());
        return StubBackend::pl_commit(q, n_keep, err);
    }
    bool pl_undo(int q, std::string& err) override {
        if (!book.on_pl_undo(q, err)) return false;
        state = text(book.history());
        return StubBackend::pl_undo(q, err);
    }
    bool images(FrontImages& im, std::string& err) override {
        book.on_abandon();
        book.set_request_images(images_below(im.images, im.n));
        return StubBackend::images(im, err);
    }
    bool session(SessionOp op, int64_t a, int64_t b, std::string& err) override {
        calls.push_back("session " + std::to_string((int) op) + " " + std::to_string(b));
        if (ends_window(op)) book.on_abandon();   // (as the engine's front stage does)
        switch (op) {
        case SessionOp::Reset: book.on_reset(); state.clear(); return true;
        case SessionOp::Abandon: book.on_abandon(); return true;
        case SessionOp::Drop: book.drop_image(a, b); return true;   // (the window and the session stay)
        case SessionOp::Steer: book.set_steer(a != 0); return true;
        case SessionOp::Snapshot:
            if (!book.check(a, b, "a checkpoint", err)) return false;
            book.put_part(a, b, Part{book.history(), book.images_held()});
            return true;
        case SessionOp::Restore: {
            const Part* p = book.find_part(a, b);
            if (p == nullptr) { err = "no checkpoint"; return false; }
            state = text(p->ids);
            book.set_history(p->ids, p->imgs);
            book.set_steer(book.steer_of(a, p->ids, p->imgs));   // the switch the restored state was computed with
            return true;
        }
        case SessionOp::Park: {
            if (!book.check(a, b, "a park", err)) return false;
            Image im{book.history(), book.images_held(), book.parts_on(b, [&](const Part& p) { return key_with(p, book.steer()); })};
            book.put_image(a, b, std::move(im));
            return true;
        }
        case SessionOp::Unpark: {
            auto im = book.take_image(a, b);
            if (!im) { err = "no parked copy"; return false; }
            book.set_history(im->ids, im->imgs);
            const bool steer = book.steer_of(a, im->ids, im->imgs);   // the switch the parked state was computed with
            book.set_steer(steer);
            state = text(im->ids);
            for (Part& p : im->parts) {
                const int64_t k = key_with(p, steer), n = (int64_t) p.ids.size();
                book.put_part(k, n, std::move(p));
            }
            return true;
        }
        default: err = "refused"; return false;
        }
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
