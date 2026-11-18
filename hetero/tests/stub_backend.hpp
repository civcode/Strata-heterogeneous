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
};

/// The front stage's session bookkeeping as the engine's front backend uses it, with a string for the session state:
/// a checkpoint is the token history it was taken at, an image that history with its checkpoints.
struct BookBackend : StubBackend {
    struct Image { std::vector<int32_t> ids; std::vector<std::string> parts; };
    FrontSessionBook<std::string, Image> book{3, 2};
    std::string state;   // what the "session" holds: the tokens, as text
    static std::string text(const std::vector<int32_t>& ids) {
        std::string t;
        for (int32_t v : ids) t += std::to_string(v) + ",";
        return t;
    }
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
    bool session(SessionOp op, int64_t a, int64_t b, std::string& err) override {
        calls.push_back("session " + std::to_string((int) op) + " " + std::to_string(b));
        if (ends_window(op)) book.on_abandon();   // (as the engine's front stage does)
        switch (op) {
        case SessionOp::Reset: book.on_reset(); state.clear(); return true;
        case SessionOp::Abandon: book.on_abandon(); return true;
        case SessionOp::Drop: book.drop_image(a, b); return true;   // (the window and the session stay)
        case SessionOp::Snapshot:
            if (!book.check(a, b, "a checkpoint", err)) return false;
            book.put_part(a, b, std::string(state));
            return true;
        case SessionOp::Restore: {
            const std::string* p = book.find_part(a, b);
            if (p == nullptr) { err = "no checkpoint"; return false; }
            state = *p;
            std::vector<int32_t> ids;
            for (size_t i = 0; i < p->size();) { const size_t c = p->find(',', i); ids.push_back(std::stoi(p->substr(i, c - i))); i = c + 1; }
            book.set_history(std::move(ids));
            return true;
        }
        case SessionOp::Park: {
            if (!book.check(a, b, "a park", err)) return false;
            const auto& h = book.history();
            Image im{h, book.parts_on(h, b, [&](const std::string& s) {
                         std::vector<int32_t> ids;
                         for (size_t i = 0; i < s.size();) { const size_t c = s.find(',', i); ids.push_back(std::stoi(s.substr(i, c - i))); i = c + 1; }
                         return token_key(ids.data(), (int64_t) ids.size());
                     })};
            book.put_image(a, b, std::move(im));
            return true;
        }
        case SessionOp::Unpark: {
            auto im = book.take_image(a, b);
            if (!im) { err = "no parked copy"; return false; }
            book.set_history(im->ids);
            state = text(im->ids);
            for (std::string& p : im->parts) {
                std::vector<int32_t> ids;
                for (size_t i = 0; i < p.size();) { const size_t c = p.find(',', i); ids.push_back(std::stoi(p.substr(i, c - i))); i = c + 1; }
                book.put_part(token_key(ids.data(), (int64_t) ids.size()), (int64_t) ids.size(), std::move(p));
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
