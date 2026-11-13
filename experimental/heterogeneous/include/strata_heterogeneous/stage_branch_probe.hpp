#pragma once

// M2.15 opt-in native layer-5 router/expert and layer-6 GDN snapshots.
// No new GPU work in an ordinary inference window.
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

class StageBranchRecorder {
public:
    bool configure(core::Verifier& verifier, const Plan& plan,
                   size_t index, std::string& error) {
        const char* raw = std::getenv("STRATA_HETERO_BRANCH_PROBE_PREFIX");
        if (raw == nullptr || !*raw || index >= plan.stages.size() ||
            plan.stages[index].first_layer != 0) return true;
        prefix_ = raw;
        const std::filesystem::path path(prefix_);
        if (prefix_.size() > 4096 || !path.is_absolute() ||
            prefix_.find('\n') != std::string::npos ||
            prefix_.find('\r') != std::string::npos ||
            !std::filesystem::is_directory(path.parent_path())) {
            error = "branch probe prefix must be absolute in an existing directory";
            return false;
        }
        for (const auto& part : path) if (part == "." || part == "..") {
            error = "branch probe prefix must not include dot segments";
            return false;
        }
        if (index != 0 || plan.total_layers != 48 ||
            plan.stages[index].end_layer <= 6 ||
            plan.geometry.n_embd != 2560) {
            error = "branch probe requires full model and stage zero containing layers 5 and 6";
            return false;
        }
        stage_ = static_cast<uint32_t>(index);
        n_layers_ = plan.total_layers;
        verifier.set_stage_branch_probe(true);
        enabled_ = true;
        return true;
    }

    bool record(const core::Verifier& verifier, const Plan& plan,
                const WindowSpec& window, uint64_t request,
                uint64_t window_id, std::string& error) const {
        if (!enabled_) return true;
        if (window.token_ids.size() != 1 || window.token_ids[0] < 0 ||
            window.pos0 < 0 || request == 0 || window_id == 0 ||
            plan.total_layers != n_layers_ || stage_ != 0) {
            error = "branch probe requires one completed valid token window";
            return false;
        }
        core::StageBranchProbe snapshot;
        if (!verifier.copy_stage_branch_probe(snapshot, error)) return false;
        const size_t N = snapshot.n_embd, NE = snapshot.n_expert, K = snapshot.k;
        const size_t C = snapshot.conv_channels, HV = snapshot.value_heads, ZV = snapshot.value_dim;
        // Strict versioned Q2_0 geometry limits and a 1-MiB diagnostic cap.
        const size_t expected = NE + K + N + K*N + N + C + C + HV + HV + ZV + ZV + N;
        if (N != 2560 || NE != 512 || K != 10 || C == 0 || C > 16384 ||
            HV == 0 || HV > 1024 || ZV == 0 || ZV > 16384 ||
            snapshot.data.size() != expected || snapshot.ids.size() != K ||
            72 + (1 + K + expected) * sizeof(uint32_t) > (1u << 20)) {
            error = "branch probe model shape, component sizes or capture cap invalid";
            return false;
        }
        for (int32_t id : snapshot.ids) if (id < 0 || static_cast<size_t>(id) >= NE) {
            error = "branch probe contains out-of-vocabulary expert ID";
            return false;
        }
        for (float value : snapshot.data) if (!std::isfinite(value)) {
            error = "branch probe has nonfinite FP32 data";
            return false;
        }
        std::vector<uint8_t> bytes;
        bytes.reserve(72 + (1 + K + expected) * sizeof(uint32_t));
        // STB0/v1: 72-byte LE header, token i32, K selected expert IDs,
        // then 12 fields in the fixed declared order, FP32 LE.
        // FNV1a checksum excludes byte offsets 64..67.
        add32(bytes, 0x30425453u); // "STB0"
        add32(bytes, 1);
        add32(bytes, stage_);
        add32(bytes, static_cast<uint32_t>(N));
        add32(bytes, static_cast<uint32_t>(NE));
        add32(bytes, static_cast<uint32_t>(K));
        add32(bytes, static_cast<uint32_t>(C));
        add32(bytes, static_cast<uint32_t>(HV));
        add64(bytes, request);
        add64(bytes, window_id);
        add64(bytes, static_cast<uint64_t>(window.pos0));
        add64(bytes, plan.fingerprint());
        add32(bytes, 0);  // checksum
        add32(bytes, static_cast<uint32_t>(ZV));
        add32(bytes, static_cast<uint32_t>(window.token_ids[0]));
        for (int32_t id : snapshot.ids) add32(bytes, static_cast<uint32_t>(id));
        for (float val : snapshot.data) {
            uint32_t bits;
            std::memcpy(&bits, &val, sizeof(bits));
            add32(bytes, bits);
        }
        uint32_t digest = 2166136261u;
        for (size_t i = 0; i < bytes.size(); ++i) {
            if (i >= 64 && i < 68) continue;
            digest = (digest ^ bytes[i]) * 16777619u;
        }
        for (unsigned j = 0; j < 4; ++j)
            bytes[64 + j] = static_cast<uint8_t>(digest >> (8 * j));
        const std::string name = prefix_ + ".stage0.w" +
                                 std::to_string(window_id) + ".stb0";
        const int fd = ::open(name.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
        if (fd < 0) {
            error = "branch probe record already exists or cannot open: " +
                    name + ": " + std::strerror(errno);
            return false;
        }
        bool ok = true;
        for (size_t offset = 0; offset < bytes.size();) {
            const ssize_t n = ::write(fd, bytes.data() + offset, bytes.size() - offset);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { ok = false; break; }
            offset += static_cast<size_t>(n);
        }
        if (::close(fd) != 0) ok = false;
        if (!ok) {
            (void)::unlink(name.c_str());
            error = "branch probe record write failed: " + name;
            return false;
        }
        return true;
    }

private:
    static void add32(std::vector<uint8_t>& out, uint32_t x) {
        for (unsigned i = 0; i < 4; ++i)
            out.push_back(static_cast<uint8_t>(x >> (8 * i)));
    }
    static void add64(std::vector<uint8_t>& out, uint64_t x) {
        for (unsigned i = 0; i < 8; ++i)
            out.push_back(static_cast<uint8_t>(x >> (8 * i)));
    }
    std::string prefix_;
    uint32_t stage_ = 0, n_layers_ = 0;
    bool enabled_ = false;
};

} // namespace strata::heterogeneous
