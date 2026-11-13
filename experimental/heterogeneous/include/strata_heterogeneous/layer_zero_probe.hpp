#pragma once

// M2.12 opt-in layer-zero arithmetic input/output probe (CUDA/HIP native
// worker only). No GPU work or file I/O when disabled by the supervisor.
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

class LayerZeroProbeRecorder {
public:
    bool configure(core::Verifier& verifier, const Plan& plan,
                   size_t index, std::string& error) {
        const char* raw = std::getenv("STRATA_HETERO_L0_PROBE_PREFIX");
        if (raw == nullptr || !*raw || index >= plan.stages.size() ||
            plan.stages[index].first_layer != 0) return true;
        prefix_ = raw;
        const std::filesystem::path path(prefix_);
        if (prefix_.size() > 4096 || !path.is_absolute() ||
            prefix_.find('\n') != std::string::npos ||
            prefix_.find('\r') != std::string::npos ||
            !std::filesystem::is_directory(path.parent_path())) {
            error = "layer-zero probe prefix must be absolute with an existing parent";
            return false;
        }
        for (const auto& segment : path)
            if (segment == "." || segment == "..") {
                error = "layer-zero probe may not use dot segments";
                return false;
            }
        stage_ = static_cast<uint32_t>(index);
        n_layers_ = plan.total_layers;
        verifier.set_layer_zero_arithmetic_probe(true); // BEFORE init/capture
        enabled_ = true;
        return true;
    }

    bool record(const core::Verifier& verifier, const Plan& plan,
                const WindowSpec& window, uint64_t request,
                uint64_t window_id, std::string& error) const {
        if (!enabled_) return true;
        if (window.token_ids.size() != 1 || window.token_ids[0] < 0 ||
            window.pos0 < 0 || !request || !window_id ||
            plan.total_layers != n_layers_ ||
            plan.stages[stage_].first_layer != 0) {
            error = "layer-zero arithmetic probe supports a single completed token row";
            return false;
        }
        core::LayerZeroArithmetic snapshot;
        if (!verifier.copy_layer_zero_arithmetic(snapshot, error)) return false;
        const uint32_t N = snapshot.n_embd;
        const uint32_t C = snapshot.qkv_channels;
        const size_t Q = (static_cast<size_t>(N) / 32) * 36;
        if (N == 0 || N > 16384 || N % 32 != 0 || C == 0 || C > 65536 ||
            snapshot.embedded.size() != N || snapshot.mixed.size() != N ||
            snapshot.q8_bytes.size() != Q || snapshot.qkv.size() != C ||
            N != plan.geometry.n_embd) {
            error = "layer-zero arithmetic probe has invalid geometry or size";
            return false;
        }
        for (const auto* arr : {&snapshot.embedded, &snapshot.mixed, &snapshot.qkv})
            for (const float value : *arr)
                if (!std::isfinite(value)) {
                    error = "nonfinite layer-zero arithmetic probe value";
                    return false;
                }
        // STL0/v1: LE 72-byte header with FNV1a checksum at 64..67,
        // followed by one i32 token, N embedding f32, N mixed f32,
        // (N/32)*36 Q8_1 raw activation bytes, C projected QKV f32.
        std::vector<uint8_t> bytes;
        const size_t expected = 72 + 4 + (2 * static_cast<size_t>(N) + C) * 4 + Q;
        if (expected > (1u << 20)) {
            error = "layer-zero arithmetic record exceeds 1 MiB";
            return false;
        }
        bytes.reserve(expected);
        add32(bytes, 0x304c5453u); // "STL0"
        add32(bytes, 1);
        add32(bytes, stage_);
        add32(bytes, N);
        add32(bytes, C);
        add32(bytes, static_cast<uint32_t>(Q));
        add32(bytes, 1);
        add32(bytes, n_layers_);
        add64(bytes, request);
        add64(bytes, window_id);
        add64(bytes, static_cast<uint64_t>(window.pos0));
        add64(bytes, plan.fingerprint());
        add32(bytes, 0); // checksum
        add32(bytes, 0); // reserved
        add32(bytes, static_cast<uint32_t>(window.token_ids[0]));
        auto add_floats = [&](const std::vector<float>& data) {
            for (float value : data) {
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                add32(bytes, bits);
            }
        };
        add_floats(snapshot.embedded);
        add_floats(snapshot.mixed);
        bytes.insert(bytes.end(), snapshot.q8_bytes.begin(), snapshot.q8_bytes.end());
        add_floats(snapshot.qkv);
        if (bytes.size() != expected) {
            error = "layer-zero arithmetic encoded length mismatch";
            return false;
        }
        uint32_t hash = 2166136261u;
        for (size_t i = 0; i < bytes.size(); ++i) {
            if (i >= 64 && i < 68) continue;
            hash = (hash ^ bytes[i]) * 16777619u;
        }
        for (unsigned j = 0; j < 4; ++j)
            bytes[64 + j] = static_cast<uint8_t>(hash >> (j * 8));
        const std::string filename = prefix_ + ".stage" + std::to_string(stage_) +
                                     ".w" + std::to_string(window_id) + ".stl0";
        const int fd = ::open(filename.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
        if (fd < 0) {
            error = "layer-zero arithmetic record exists or cannot open: " +
                    filename + ": " + std::strerror(errno);
            return false;
        }
        bool okay = true;
        for (size_t offset = 0; offset < bytes.size();) {
            const ssize_t wrote = ::write(fd, bytes.data() + offset, bytes.size() - offset);
            if (wrote < 0 && errno == EINTR) continue;
            if (wrote <= 0) { okay = false; break; }
            offset += static_cast<size_t>(wrote);
        }
        if (::close(fd) != 0) okay = false;
        if (!okay) {
            (void)::unlink(filename.c_str());
            error = "failed writing layer-zero arithmetic probe: " + filename;
            return false;
        }
        return true;
    }

private:
    static void add32(std::vector<uint8_t>& out, uint32_t value) {
        for (unsigned i = 0; i < 4; ++i)
            out.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
    static void add64(std::vector<uint8_t>& out, uint64_t value) {
        for (unsigned i = 0; i < 8; ++i)
            out.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
    std::string prefix_;
    uint32_t stage_ = 0;
    uint32_t n_layers_ = 0;
    bool enabled_ = false;
};

} // namespace strata::heterogeneous
