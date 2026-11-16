#pragma once
// Application wire protocol for the heterogeneous coordinator. Internal worker
// commit/ACK frames are not visible to the vanilla Python server.
#include "strata/core/serve_request_summary.hpp"
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <istream>
#include <limits>
#include <optional>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace strata::heterogeneous::vanilla {

inline uint32_t number(const std::string& s) {
    uint32_t v = 0;
    auto r = std::from_chars(s.data(), s.data()+s.size(), v);
    if (s.empty() || r.ec != std::errc{} || r.ptr != s.data()+s.size())
        throw std::invalid_argument("GEN: invalid integer");
    return v;
}

struct Request { uint32_t max_new = 0; std::vector<int32_t> ids; };

inline void check_key(const std::string& key, const std::string& value) {
    // Greedy generation: seed and penalty anchor cannot affect argmax when
    // penalties are disabled. ckpt=0 requests the already uncached behavior.
    if (key == "seed" || key == "pen_from" || key == "penalty_last_n") {
        (void)number(value); return;
    }
    if (key == "ckpt" && value == "0") return;
    if (key == "top_k" && (value == "1" || value == "64" || value == "0")) return;
    if (key == "temperature" || key == "min_p" || key == "top_p" ||
        key == "penalty_repeat" || key == "penalty_freq" || key == "penalty_present") {
        size_t used = 0;
        double v = std::stod(value, &used);
        const double neutral = (key == "top_p" || key == "penalty_repeat") ? 1.0 : 0.0;
        if (used == value.size() && std::isfinite(v) && v == neutral) return;
    }
    throw std::invalid_argument("GEN: unsupported key or value " + key);
}

inline Request parse_gen(const std::string& line, uint32_t context, uint32_t vocab, uint32_t max_output) {
    std::istringstream in(line);
    std::string op, word;
    if (!(in >> op >> word) || op != "GEN") throw std::invalid_argument("expected GEN");
    Request r;
    r.max_new = number(word);
    if (!r.max_new || r.max_new > max_output) throw std::invalid_argument("GEN: completion budget exceeds engine limit");
    std::vector<std::string> seen;
    bool got_ids = false;
    while (in >> word) {
        if (got_ids) throw std::invalid_argument("GEN: unexpected trailing argument");
        auto eq = word.find('=');
        if (eq != std::string::npos) {
            const auto key = word.substr(0, eq);
            for (const auto& prior : seen) if (prior == key) throw std::invalid_argument("GEN: duplicate key " + key);
            seen.push_back(key);
            check_key(key, word.substr(eq+1));
            continue;
        }
        got_ids = true;
        std::istringstream tokens(word);
        std::string item;
        while (std::getline(tokens, item, ',')) {
            const auto t = number(item);
            if (t >= vocab || t > static_cast<uint32_t>(INT32_MAX) || r.ids.size() >= context)
                throw std::invalid_argument("GEN: prompt token exceeds engine limit");
            r.ids.push_back(static_cast<int32_t>(t));
        }
        if (word.back() == ',') throw std::invalid_argument("GEN: invalid token list");
    }
    if (r.ids.empty() || uint64_t(r.ids.size()) + r.max_new > context)
        throw std::invalid_argument("GEN: prompt and completion exceed context");
    return r;
}

// One reader owns stdin, including STOP/QUIT during execution. Do not mix
// poll() with std::cin buffering; preserve partial lines between checkpoints.
class Input {
public:
    explicit Input(size_t cap, int fd = STDIN_FILENO) : cap_(cap), fd_(fd) {}
    std::optional<std::string> line(bool wait) {
        for (;;) {
            const auto nl = pending_.find('\n', scan_);
            if (nl != std::string::npos) {
                if (nl > cap_) throw std::runtime_error("serve command exceeds byte cap");
                auto out = pending_.substr(0, nl);
                pending_.erase(0, nl+1); scan_ = 0;
                return out;
            }
            scan_ = pending_.size();
            if (scan_ > cap_) throw std::runtime_error("serve command exceeds byte cap");
            if (eof_) {
                if (!pending_.empty()) throw std::runtime_error("incomplete serve command");
                return std::nullopt;
            }
            pollfd p{fd_, POLLIN, 0};
            int rc;
            do { rc = ::poll(&p, 1, wait ? -1 : 0); } while (rc < 0 && errno == EINTR);
            if (rc < 0 || (p.revents & (POLLERR | POLLNVAL))) throw std::runtime_error("serve input poll failed");
            if (!rc) return std::nullopt;
            char buf[4096];
            ssize_t n;
            do { n = ::read(fd_, buf, sizeof buf); } while (n < 0 && errno == EINTR);
            if (n < 0) throw std::runtime_error("serve input read failed");
            if (n == 0) eof_ = true;
            else pending_.append(buf, static_cast<size_t>(n));
        }
    }
    enum class Control { proceed, stop, quit, eof };
    Control checkpoint() {
        auto cmd = line(false);
        if (!cmd) return eof_ ? Control::eof : Control::proceed;
        if (*cmd == "STOP") return Control::stop;
        if (*cmd == "QUIT") return Control::quit;
        throw std::runtime_error("unexpected command during GEN");
    }
private:
    size_t cap_, scan_ = 0;
    int fd_;
    bool eof_ = false;
    std::string pending_;
};

inline void print_progress(std::FILE* out, int64_t read, int64_t total, double ms) {
    std::fprintf(out, "PP %lld %lld %.0f %.1f\n", (long long)read, (long long)total, ms,
                 ms > 0 ? 1000.0 * read / ms : 0.0);
    std::fflush(out);
}

inline void print_done(std::FILE* out, const strata::core::ServeRequestSummary& s, const char* finish) {
    // Vanilla's backwards-compatible DONE prefix: later cache/IO counters
    // are omitted because STFR does not carry those counters across owners.
    std::fprintf(out, "DONE %lld %lld %.1f %.1f %s %lld %lld %lld\n",
                 (long long)s.generated, (long long)s.prompt_tokens, s.prompt_ms, s.decode_ms, finish,
                 (long long)s.drafts_accepted, (long long)s.drafts_offered, (long long)s.reused);
    std::fflush(out);
}

} // namespace strata::heterogeneous::vanilla
