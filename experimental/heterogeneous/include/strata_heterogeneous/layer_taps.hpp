#pragma once

// Diagnostic-only native verifier layer taps for M2.11. Never included by
// the host coordinator and never active unless explicitly requested.
// The data are post-layer hc-mean residuals with the pending write applied,
// NOT the complete [R|bo|inject] handoff or a numerical reference by itself.
#include "strata_heterogeneous/host.hpp"
#include "strata/core/verify.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <string>
#include <vector>
#include <unistd.h>

namespace strata::heterogeneous {

class LayerTapRecorder {
public:
    bool configure(core::Verifier& verifier, const Plan& plan,
                   size_t stage_index, std::string& error) {
        const char* raw = std::getenv("STRATA_HETERO_LAYER_TAP_PREFIX");
        if (!raw || !*raw) return true;
        if (stage_index >= plan.stages.size()) {
            error = "tap stage index out of range"; return false;
        }
        prefix_ = raw;
        const std::filesystem::path dest(prefix_);
        if (prefix_.size() > 4096 || !dest.is_absolute() ||
            prefix_.find('\n') != std::string::npos ||
            prefix_.find('\r') != std::string::npos ||
            !std::filesystem::is_directory(dest.parent_path())) {
            error = "tap prefix must be an absolute path under an existing directory";
            return false;
        }
        for (const auto& part : dest)
            if (part == "." || part == "..") {
                error = "tap prefix may not contain dot segments"; return false;
            }
        const Stage& stage = plan.stages[stage_index];
        index_ = static_cast<uint32_t>(stage_index);
        first_ = stage.first_layer;
        width_ = plan.geometry.n_embd;
        total_ = plan.total_layers;
        count_ = (stage.first_layer < total_ - 1)
            ? std::min(stage.end_layer, total_ - 1) - first_ : 0;
        if (!count_ || count_ > 48 || width_ == 0 || width_ > 16384 ||
            total_ > 128) {
            error = "no bounded post-layer taps available for this stage";
            return false;
        }
        std::vector<int> layers;
        for (uint32_t layer = first_; layer < first_ + count_; ++layer)
            layers.push_back(static_cast<int>(layer));
        verifier.set_taps(layers); // before Verifier::init/capture_all
        active_ = true;
        return true;
    }

    bool record(core::Verifier& verifier, const Plan& plan,
                const WindowSpec& spec, uint64_t request_id, uint64_t window_id,
                std::string& error) const {
        if (!active_) return true;
        const uint32_t rows = static_cast<uint32_t>(spec.token_ids.size());
        if (!rows || rows > 8 || spec.pos0 < 0 || !request_id || !window_id ||
            plan.total_layers != total_ || plan.geometry.n_embd != width_) {
            error = "invalid layer-tap window or model identity"; return false;
        }
        const size_t floats = static_cast<size_t>(rows) * count_ * width_;
        // A bounded diagnostic: max 8 rows * 48 taps * 16384 f32.
        if (floats > (32u << 20) / sizeof(float)) {
            error = "layer-tap diagnostic exceeds 32 MiB limit"; return false;
        }
        std::vector<float> values(floats);
        if (!verifier.gather_taps_host(values.data(), 0, static_cast<int>(rows), error))
            return false;
        for (const float f : values) {
            if (!std::isfinite(f)) {
                error = "nonfinite post-layer tap detected"; return false;
            }
        }
        // STLT/v1 little endian: 72-byte header; rows token IDs;
        // rows*count*width FP32 (row, layer, feature) and FNV-1a checksum
        // excluding header bytes 64..67.
        std::vector<uint8_t> bytes;
        bytes.reserve(72 + static_cast<size_t>(rows) * 4 + floats * 4);
        put32(bytes, 0x544c5453u); // 'STLT'
        put32(bytes, 1);
        put32(bytes, index_);
        put32(bytes, first_);
        put32(bytes, count_);
        put32(bytes, width_);
        put32(bytes, rows);
        put32(bytes, total_);
        put64(bytes, request_id);
        put64(bytes, window_id);
        put64(bytes, static_cast<uint64_t>(spec.pos0));
        put64(bytes, plan.fingerprint());
        put32(bytes, 0); // checksum offset 64
        put32(bytes, 0); // reserved
        for (const int32_t token : spec.token_ids)
            put32(bytes, static_cast<uint32_t>(token));
        for (const float f : values) {
            uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(f), "FP32 tap needed");
            std::memcpy(&bits, &f, sizeof(bits));
            put32(bytes, bits);
        }
        uint32_t digest = 2166136261u;
        for (size_t i = 0; i < bytes.size(); ++i) {
            if (i >= 64 && i < 68) continue;
            digest = (digest ^ bytes[i]) * 16777619u;
        }
        for (unsigned j = 0; j < 4; ++j)
            bytes[64 + j] = static_cast<uint8_t>(digest >> (j * 8));

        const std::string name = prefix_ + ".stage" + std::to_string(index_) +
                                 ".w" + std::to_string(window_id) + ".stlt";
        const int fd = ::open(name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) {
            error = "cannot create unique tap record: " + name +
                    ": " + std::strerror(errno);
            return false;
        }
        bool ok = true;
        for (size_t off = 0; off < bytes.size();) {
            const ssize_t n = ::write(fd, bytes.data() + off, bytes.size() - off);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { ok = false; break; }
            off += static_cast<size_t>(n);
        }
        if (::close(fd) != 0) ok = false;
        if (!ok) {
            (void) ::unlink(name.c_str());
            error = "cannot complete layer-tap record: " + name;
            return false;
        }
        return true;
    }

private:
    static void put32(std::vector<uint8_t>& out, uint32_t value) {
        for (unsigned i = 0; i < 4; ++i)
            out.push_back(static_cast<uint8_t>(value >> (i * 8)));
    }
    static void put64(std::vector<uint8_t>& out, uint64_t value) {
        for (unsigned i = 0; i < 8; ++i)
            out.push_back(static_cast<uint8_t>(value >> (i * 8)));
    }

    std::string prefix_;
    uint32_t index_ = 0, first_ = 0, count_ = 0, width_ = 0, total_ = 0;
    bool active_ = false;
};

} // namespace strata::heterogeneous
