#pragma once

// The front stage's session bookkeeping for a mixed CUDA/HIP split (.ai/SEAM.md).
//
// The host keeps vanilla's checkpoints and parked conversations for its own layers.  The front stage repeats each
// save and restore on its own layers and keeps the result here, under the token prefix it belongs to: the host sends
// (token_key(ids, n), n) with every session operation, and the front stage checks it against the tokens its own
// session holds before it saves or restores anything.  A difference means the two processes no longer hold the same
// conversation, which would otherwise show up only as a wrong continuation.
//
// `Part` is a checkpoint of the front stage's session (vanilla's ConversationCheckpoint), `Image` a parked copy of it
// (vanilla's SavedConversation).  The CPU tests use plain types.  No GPU, engine or transport dependency.

#include <algorithm>
#include <cstdint>
#include <list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace strata::hetero {

/// FNV-1a over the 32-bit token values of `ids[0, n)` and `n`, as a positive int64 (never 0).  The host's prompt
/// buffer holds int64 tokens, its checkpoints int32 ones; both give the same key.
template <class T>
inline int64_t token_key(const T* ids, int64_t n) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint32_t v) {
        for (int k = 0; k < 4; ++k) { h ^= (v >> (8 * k)) & 0xffu; h *= 1099511628211ull; }
    };
    for (int64_t i = 0; i < n; ++i) mix((uint32_t) (int32_t) ids[i]);
    mix((uint32_t) n);
    mix((uint32_t) ((uint64_t) n >> 32));
    const int64_t k = (int64_t) (h & 0x7fffffffffffffffull);
    return k == 0 ? 1 : k;
}

template <class Part, class Image>
class FrontSessionBook {
public:
    /// `max_parts` / `max_images`: how many of each are kept (least recently used goes first).
    FrontSessionBook(size_t max_parts, size_t max_images) : max_parts_(std::max<size_t>(max_parts, 1)), max_images_(max_images) {}

    // ---- the tokens the session holds: every operation that changes the session reports here
    void on_reset() { hist_.clear(); window_ = false; }
    /// A prompt chunk at `pos0`: refused unless it continues the session exactly.
    bool on_prefill(const int32_t* tokens, int64_t n, int64_t pos0, std::string& err) {
        if (!continues(pos0, "prompt chunk", err)) return false;
        hist_.insert(hist_.end(), tokens, tokens + n);
        return true;
    }
    /// A window at `pos0` (kept only by on_commit).
    bool on_run(const int32_t* tokens, int T, int64_t pos0, std::string& err) {
        if (!continues(pos0, "window", err)) return false;
        win_.assign(tokens, tokens + T);
        window_ = true;
        return true;
    }
    void on_commit(int n_keep) {
        if (!window_) return;
        hist_.insert(hist_.end(), win_.begin(), win_.begin() + std::min<int64_t>(n_keep, (int64_t) win_.size()));
        window_ = false;
    }
    void on_abandon() { window_ = false; }
    const std::vector<int32_t>& history() const { return hist_; }
    void set_history(std::vector<int32_t> ids) { hist_ = std::move(ids); window_ = false; }

    /// The session holds exactly the `n` tokens the host's `key` names.
    bool check(int64_t key, int64_t n, const char* what, std::string& err) const {
        if (n < 0 || (int64_t) hist_.size() != n) {
            err = std::string(what) + " at " + std::to_string((long long) n) + " tokens, but the front stage holds " +
                  std::to_string((long long) hist_.size());
            return false;
        }
        if (token_key(hist_.data(), n) != key) {
            err = std::string(what) + " at " + std::to_string((long long) n) +
                  " tokens: the front stage holds other tokens than the host";
            return false;
        }
        return true;
    }

    // ---- checkpoints of the front stage's session (vanilla's checkpoint chain, one part per host checkpoint)
    void put_part(int64_t key, int64_t n, Part&& p) { put(parts_, max_parts_, key, n, std::move(p)); }
    Part* find_part(int64_t key, int64_t n) {
        for (auto it = parts_.begin(); it != parts_.end(); ++it)
            if (it->key == key && it->n == n) {
                parts_.splice(parts_.end(), parts_, it);   // most recently used last
                return &parts_.back().v;
            }
        return nullptr;
    }
    /// Copies of the parts that are prefixes of `ids[0, n)` (what a parked image of that conversation holds).
    template <class Key>
    std::vector<Part> parts_on(const std::vector<int32_t>& ids, int64_t n, Key key_of_part) const {
        std::vector<Part> out;
        for (const auto& e : parts_)
            if (e.n <= n && e.n <= (int64_t) ids.size() && token_key(ids.data(), e.n) == e.key && key_of_part(e.v) == e.key)
                out.push_back(e.v);
        return out;
    }
    size_t parts() const { return parts_.size(); }

    // ---- parked copies (vanilla's conversation cache): put when the host parks, taken when it restores
    void put_image(int64_t key, int64_t n, Image&& im) { if (max_images_ > 0) put(images_, max_images_, key, n, std::move(im)); }
    std::optional<Image> take_image(int64_t key, int64_t n) {
        for (auto it = images_.begin(); it != images_.end(); ++it)
            if (it->key == key && it->n == n) {
                std::optional<Image> v(std::move(it->v));
                images_.erase(it);
                return v;
            }
        return std::nullopt;
    }
    /// The host's conversation cache no longer holds the copy of these tokens: drop ours.  False when there was none.
    bool drop_image(int64_t key, int64_t n) {
        const size_t before = images_.size();
        images_.remove_if([&](const Entry<Image>& e) { return e.key == key && e.n == n; });
        return images_.size() != before;
    }
    size_t images() const { return images_.size(); }
    void clear_stores() { parts_.clear(); images_.clear(); }

private:
    template <class V>
    struct Entry { int64_t key; int64_t n; V v; };
    template <class V>
    static void put(std::list<Entry<V>>& l, size_t cap, int64_t key, int64_t n, V&& v) {
        l.remove_if([&](const Entry<V>& e) { return e.key == key && e.n == n; });   // the same tokens: the newer state
        l.push_back(Entry<V>{key, n, std::move(v)});
        while (l.size() > cap) l.pop_front();
    }
    bool continues(int64_t pos0, const char* what, std::string& err) const {
        if (pos0 == (int64_t) hist_.size()) return true;
        err = std::string("a ") + what + " at position " + std::to_string((long long) pos0) + ", but the front stage holds " +
              std::to_string((long long) hist_.size()) + " tokens (the two sessions are out of step)";
        return false;
    }
    size_t max_parts_, max_images_;
    std::vector<int32_t> hist_, win_;
    bool window_ = false;
    std::list<Entry<Part>> parts_;
    std::list<Entry<Image>> images_;
};

} // namespace strata::hetero
