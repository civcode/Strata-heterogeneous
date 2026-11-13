#pragma once

// M2.14: small, opt-in stage-zero four-phase FP32 snapshots; one token row.
// No extra allocations, memcpy nodes or file I/O in ordinary inference.
#include "strata_heterogeneous/host.hpp"
#include "strata/core/verify.hpp"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <string>
#include <vector>
#include <unistd.h>

namespace strata::heterogeneous {

class Stage0PhaseRecorder {
public:
    bool configure(core::Verifier& verifier, const Plan& plan,
                   size_t index, std::string& error) {
        const char* raw = std::getenv("STRATA_HETERO_STAGE0_PHASE_PREFIX");
        if (!raw || !*raw || index >= plan.stages.size() ||
            plan.stages[index].first_layer != 0)
            return true;
        prefix_ = raw;
        const std::filesystem::path p(prefix_);
        if (prefix_.size() > 4096 || !p.is_absolute() ||
            prefix_.find('\n') != std::string::npos ||
            prefix_.find('\r') != std::string::npos ||
            !std::filesystem::is_directory(p.parent_path())) {
            error = "stage0 phase prefix must be absolute with existing parent";
            return false;
        }
        for (const auto& segment : p)
            if (segment == "." || segment == "..") {
                error = "stage0 phase prefix may not have dot segments";
                return false;
            }
        stage_ = static_cast<uint32_t>(index);
        n_layers_ = plan.total_layers;
        width_ = plan.geometry.n_embd;
        if (stage_ != 0 || n_layers_ != 48 || width_ != 2560 ||
            plan.stages[index].end_layer < core::Verifier::kStage0ProbeLayers) {
            error = "stage0 phase probe requires a full 12-layer first stage";
            return false;
        }
        verifier.set_stage0_phase_probe(true); // before init/graph capture
        enabled_ = true;
        return true;
    }

    bool record(const core::Verifier& verifier, const Plan& plan,
                const WindowSpec& window, uint64_t request,
                uint64_t window_id, std::string& error) const {
        if (!enabled_) return true;
        if (window.token_ids.size() != 1 || window.token_ids[0] < 0 ||
            window.pos0 < 0 || request == 0 || window_id == 0 ||
            stage_ != 0 || plan.total_layers != n_layers_ ||
            plan.geometry.n_embd != width_) {
            error = "stage0 phase probe requires a valid one-row committed window";
            return false;
        }
        std::vector<float> samples;
        if (!verifier.copy_stage0_phase_probe(samples, error)) return false;
        const size_t count = static_cast<size_t>(core::Verifier::kStage0ProbeLayers) *
                             core::Verifier::kStage0ProbePhases * width_;
        if (samples.size() != count || count * sizeof(float) > (1u << 20)) {
            error = "invalid or oversized stage0 phase probe";
            return false;
        }
        for (float f : samples)
            if (!std::isfinite(f)) {
                error = "nonfinite stage0 phase probe";
                return false;
            }

        // STP0/v1: 72-byte LE header, i32 token, [layer][phase][feature]
        // FP32 values. FNV1a checksum skips header bytes 64..67.
        std::vector<uint8_t> bytes;
        bytes.reserve(72 + 4 + count * 4);
        add32(bytes, 0x30505453u); // "STP0"
        add32(bytes, 1);
        add32(bytes, stage_);
        add32(bytes, core::Verifier::kStage0ProbeLayers);
        add32(bytes, core::Verifier::kStage0ProbePhases);
        add32(bytes, width_);
        add32(bytes, 1); // rows
        add32(bytes, n_layers_);
        add64(bytes, request);
        add64(bytes, window_id);
        add64(bytes, static_cast<uint64_t>(window.pos0));
        add64(bytes, plan.fingerprint());
        add32(bytes, 0); // checksum
        add32(bytes, 0); // reserved
        add32(bytes, static_cast<uint32_t>(window.token_ids[0]));
        for (float f : samples) {
            uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(f), "STP0 needs FP32");
            std::memcpy(&bits, &f, sizeof(bits));
            add32(bytes, bits);
        }
        if (bytes.size() != 72 + 4 + count * 4) {
            error = "stage0 phase wire size mismatch";
            return false;
        }
        uint32_t h = 2166136261u;
        for (size_t i = 0; i < bytes.size(); ++i) {
            if (i >= 64 && i < 68) continue;
            h = (h ^ bytes[i]) * 16777619u;
        }
        for (unsigned j = 0; j < 4; ++j)
            bytes[64 + j] = static_cast<uint8_t>(h >> (j * 8));

        const std::string file = prefix_ + ".stage0.w" +
                                 std::to_string(window_id) + ".stp0";
        const int fd = ::open(file.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
        if (fd < 0) {
            error = "cannot create unique stage0 phase record: " + file +
                    ": " + std::strerror(errno);
            return false;
        }
        bool okay = true;
        for (size_t off = 0; off < bytes.size();) {
            const ssize_t n = ::write(fd, bytes.data() + off, bytes.size() - off);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { okay = false; break; }
            off += static_cast<size_t>(n);
        }
        if (::close(fd) != 0) okay = false;
        if (!okay) {
            (void) ::unlink(file.c_str());
            error = "cannot complete stage0 phase record: " + file;
            return false;
        }
        return true;
    }

private:
    static void add32(std::vector<uint8_t>& out, uint32_t x) {
        for (unsigned i = 0; i < 4; ++i)
            out.push_back(static_cast<uint8_t>(x >> (i * 8)));
    }
    static void add64(std::vector<uint8_t>& out, uint64_t x) {
        for (unsigned i = 0; i < 8; ++i)
            out.push_back(static_cast<uint8_t>(x >> (i * 8)));
    }
    std::string prefix_;
    uint32_t stage_ = 0;
    uint32_t n_layers_ = 0;
    uint32_t width_ = 0;
    bool enabled_ = false;
};

} // namespace strata::heterogeneous
