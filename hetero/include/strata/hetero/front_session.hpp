#pragma once

// The front stage's session bookkeeping for a mixed CUDA/HIP split (.ai/SEAM.md).
//
// The host keeps vanilla's checkpoints and parked conversations for its own layers.  The front stage repeats each
// save and restore on its own layers and keeps the result here, under the token prefix it belongs to: the host sends
// (state_key(ids, n, ..), n) with every session operation (the tokens, the control-vector switch and the pictures below
// n), and the front stage checks it against the state its own session holds before it saves or restores anything.  A difference means the two processes no longer hold the same
// conversation, which would otherwise show up only as a wrong continuation.
//
// `Part` is a checkpoint of the front stage's session (vanilla's ConversationCheckpoint), `Image` a parked copy of it
// (vanilla's SavedConversation).  The CPU tests use plain types.  No GPU, engine or transport dependency.

#include "strata/hetero/front_stage.hpp"

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

/// A state computed with the control vector switched off is another state than the same tokens with it on (vanilla's
/// conversation cache keeps them apart by SavedConversation::cvec).  The key of a state saved with the vector off is
/// the token key mixed with a constant; with it on, or with no vector loaded, the token key itself.
inline int64_t steer_key(int64_t key, bool steer) {
    if (steer) return key;
    const int64_t k = (int64_t) (((uint64_t) key * 0x9e3779b97f4a7c15ull ^ 0x5bd1e9955bd1e995ull) & 0x7fffffffffffffffull);
    return k == 0 ? 1 : k;
}

/// Two pictures with the same `<|image_pad|>` tokens are different states (vanilla's caches compare the image keys
/// below a prefix's length with its tokens).  The key mixes in the (start, hash) of every picture that starts below
/// `n`; with none it is `key` itself, so the keys of text are unchanged.  `K`: any type with `start` and `hash`
/// (vanilla's ConversationImageKey, ImageKey, ImageSpan).
template <class K>
inline int64_t image_key(int64_t key, const std::vector<K>& imgs, int64_t n) {
    uint64_t h = (uint64_t) key;
    bool any = false;
    auto mix = [&h](uint64_t v) {
        for (int k = 0; k < 8; ++k) { h ^= (v >> (8 * k)) & 0xffu; h *= 1099511628211ull; }
    };
    for (const K& im : imgs) {
        if (im.start >= n) continue;
        any = true;
        mix((uint64_t) im.start);
        mix((uint64_t) im.hash);
    }
    if (!any) return key;
    const int64_t k = (int64_t) (h & 0x7fffffffffffffffull);
    return k == 0 ? 1 : k;
}

/// The key of a saved state: the tokens `ids[0, n)`, the control-vector switch it was computed with and the pictures
/// below `n` (what vanilla's checkpoints and parked conversations are matched by).
template <class T, class K>
inline int64_t state_key(const T* ids, int64_t n, bool steer, const std::vector<K>& imgs) {
    return image_key(steer_key(token_key(ids, n), steer), imgs, n);
}

/// The keys of `imgs` that start below `n`, as ImageKey.
template <class K>
inline std::vector<ImageKey> images_below(const std::vector<K>& imgs, int64_t n) {
    std::vector<ImageKey> v;
    for (const K& im : imgs)
        if (im.start < n) v.push_back(ImageKey{im.start, im.hash});
    return v;
}

template <class Part, class Image>
class FrontSessionBook {
public:
    /// `max_parts` / `max_images`: how many of each are kept (least recently used goes first).
    FrontSessionBook(size_t max_parts, size_t max_images) : max_parts_(std::max<size_t>(max_parts, 1)), max_images_(max_images) {}

    // ---- the tokens the session holds: every operation that changes the session reports here
    void on_reset() { hist_.clear(); imgs_.clear(); window_ = false; }
    /// The pictures of the request the host is about to read (Images): the session's own pictures below a position
    /// must be the request's for a prompt chunk or window to continue it there.
    void set_request_images(std::vector<ImageKey> keys) { req_imgs_ = std::move(keys); }
    /// A prompt chunk at `pos0`: refused unless it continues the session exactly.
    bool on_prefill(const int32_t* tokens, int64_t n, int64_t pos0, std::string& err) {
        if (!continues(pos0, "prompt chunk", err)) return false;
        hist_.insert(hist_.end(), tokens, tokens + n);
        imgs_ = images_below(req_imgs_, (int64_t) hist_.size());
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
    /// The pictures of the session's state (all start below history().size()).
    const std::vector<ImageKey>& images_held() const { return imgs_; }
    /// The control-vector switch the session's state was computed with (Steer; a Restore or Unpark sets it from the
    /// key it restored).
    void set_steer(bool on) { steer_ = on; }
    bool steer() const { return steer_; }
    /// Which switch a host key of these tokens and pictures names: true for the switch on, false for off.
    template <class K = ImageKey>
    static bool steer_of(int64_t key, const std::vector<int32_t>& ids, const std::vector<K>& imgs = {}) {
        return state_key(ids.data(), (int64_t) ids.size(), true, imgs) == key;
    }
    /// A restored state: its tokens and pictures.
    template <class K = ImageKey>
    void set_history(std::vector<int32_t> ids, const std::vector<K>& imgs = {}) {
        hist_ = std::move(ids);
        imgs_ = images_below(imgs, (int64_t) hist_.size());
        window_ = false;
    }

    /// The session holds exactly the `n` tokens the host's `key` names.
    bool check(int64_t key, int64_t n, const char* what, std::string& err) const {
        if (n < 0 || (int64_t) hist_.size() != n) {
            err = std::string(what) + " at " + std::to_string((long long) n) + " tokens, but the front stage holds " +
                  std::to_string((long long) hist_.size());
            return false;
        }
        if (state_key(hist_.data(), n, steer_, imgs_) != key) {
            err = std::string(what) + " at " + std::to_string((long long) n) +
                  " tokens: the front stage holds other tokens (or pictures, or control-vector switch) than the host";
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
    /// Copies of the parts that are prefixes of the session's state at `n` tokens (what a parked image of that
    /// conversation holds).
    template <class Key>
    std::vector<Part> parts_on(int64_t n, Key key_of_part) const {
        std::vector<Part> out;
        for (const auto& e : parts_)
            if (e.n <= n && e.n <= (int64_t) hist_.size() && state_key(hist_.data(), e.n, steer_, imgs_) == e.key &&
                key_of_part(e.v) == e.key)
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
        if (pos0 != (int64_t) hist_.size()) {
            err = std::string("a ") + what + " at position " + std::to_string((long long) pos0) + ", but the front stage holds " +
                  std::to_string((long long) hist_.size()) + " tokens (the two sessions are out of step)";
            return false;
        }
        if (images_below(req_imgs_, pos0) != imgs_) {
            err = std::string("a ") + what + " at position " + std::to_string((long long) pos0) +
                  ": the front stage's session holds other pictures than this request";
            return false;
        }
        return true;
    }
    size_t max_parts_, max_images_;
    std::vector<int32_t> hist_, win_;
    std::vector<ImageKey> req_imgs_, imgs_;   // the request's pictures (Images), the session state's
    bool window_ = false;
    bool steer_ = true;   // no vector loaded: always on (vanilla's want_cvec)
    std::list<Entry<Part>> parts_;
    std::list<Entry<Image>> images_;
};

} // namespace strata::hetero
