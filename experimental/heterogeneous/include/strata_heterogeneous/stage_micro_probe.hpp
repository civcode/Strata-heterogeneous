#pragma once

// M2.17 opt-in native shared expert / GDN convolution-history diagnostics.
// STM0/v1: 72-byte LE header, one token, 9 FP32 fields, two raw Q8_1
// activation images. No work/VRAM/files unless the supervisor opts in.
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

class StageMicroRecorder {
public:
    bool configure(core::Verifier& verifier, const Plan& plan,
                   size_t index, std::string& error) {
        const char* raw = std::getenv("STRATA_HETERO_MICRO_PROBE_PREFIX");
        if (raw == nullptr || !*raw || index >= plan.stages.size() ||
            plan.stages[index].first_layer != 0) return true;
        const std::filesystem::path p(raw);
        if (std::strlen(raw) > 4096 || !p.is_absolute() ||
            p.string().find('\n') != std::string::npos ||
            p.string().find('\r') != std::string::npos ||
            !std::filesystem::is_directory(p.parent_path())) {
            error = "microprobe prefix must be absolute with existing parent directory";
            return false;
        }
        for (const auto& part : p)
            if (part == "." || part == "..") {
                error = "microprobe prefix cannot contain dot segments";
                return false;
            }
        if (index != 0 || plan.total_layers != 48 ||
            plan.stages[index].end_layer <= 6 ||
            plan.geometry.n_embd != 2560) {
            error = "microprobe requires stage0 containing layers 5,6 and canonical geometry";
            return false;
        }
        prefix_ = raw;
        verifier.set_stage_micro_probe(true); // before Verifier::init/graph capture
        enabled_ = true;
        return true;
    }

    bool record(const core::Verifier& verifier, const Plan& plan,
                const WindowSpec& window, uint64_t request,
                uint64_t window_id, std::string& error) const {
        if (!enabled_) return true;
        if (window.token_ids.size() != 1 || window.token_ids[0] < 0 ||
            window.pos0 < 0 || !request || !window_id ||
            plan.total_layers != 48) {
            error = "microprobe requires a single completed token row";
            return false;
        }
        core::StageMicroProbe probe;
        if (!verifier.copy_stage_micro_probe(probe, error)) return false;
        const size_t N = probe.n_embd, F = probe.n_ff, C = probe.conv_channels;
        const size_t D = probe.conv_kernel;
        if (N != 2560 || F == 0 || F > 65536 || F % 32 != 0 ||
            C == 0 || C > 16384 || D != 4) {
            error = "microprobe requires canonical FF/GDN geometry";
            return false;
        }
        const size_t q8in = N / 32 * 36, q8sw = F / 32 * 36;
        const size_t nfloats = N + F + F + N + 1 + N + C + C * (D - 1) + C;
        const size_t payload = 72 + 4 + nfloats * 4 + q8in + q8sw;
        if (payload > (4u << 20) || probe.data.size() != nfloats ||
            probe.input_q8.size() != q8in ||
            probe.swiglu_q8.size() != q8sw) {
            error = "microprobe component lengths exceed bounds or mismatch";
            return false;
        }
        for (float v : probe.data)
            if (!std::isfinite(v)) {
                error = "microprobe contains nonfinite FP32";
                return false;
            }
        std::vector<uint8_t> bytes;
        bytes.reserve(payload);
        add32(bytes, 0x304d5453u);  // STM0
        add32(bytes, 1);
        add32(bytes, 0); // stage0
        add32(bytes, (uint32_t) N);
        add32(bytes, (uint32_t) F);
        add32(bytes, (uint32_t) C);
        add32(bytes, (uint32_t) D);
        add32(bytes, 48);
        add64(bytes, request);
        add64(bytes, window_id);
        add64(bytes, (uint64_t) window.pos0);
        add64(bytes, plan.fingerprint());
        add32(bytes, 0); // checksum at byte 64
        add32(bytes, 0); // reserved
        add32(bytes, (uint32_t) window.token_ids[0]);
        for (float f : probe.data) {
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(bits));
            add32(bytes, bits);
        }
        bytes.insert(bytes.end(), probe.input_q8.begin(), probe.input_q8.end());
        bytes.insert(bytes.end(), probe.swiglu_q8.begin(), probe.swiglu_q8.end());
        if (bytes.size() != payload) {
            error = "microprobe encoded length mismatch";
            return false;
        }
        uint32_t checksum = 2166136261u;
        for (size_t i = 0; i < bytes.size(); ++i) {
            if (i >= 64 && i < 68) continue;
            checksum = (checksum ^ bytes[i]) * 16777619u;
        }
        for (unsigned i = 0; i < 4; ++i)
            bytes[64 + i] = (uint8_t) (checksum >> (8 * i));
        const std::string name = prefix_ + ".stage0.w" +
                                 std::to_string(window_id) + ".stm0";
        const int fd = ::open(name.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
        if (fd < 0) {
            error = "microprobe exclusive creation failed: " + name +
                    ": " + std::strerror(errno);
            return false;
        }
        bool ok = true;
        for (size_t offset = 0; offset < bytes.size();) {
            const ssize_t written = ::write(fd, bytes.data() + offset,
                                            bytes.size() - offset);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) { ok = false; break; }
            offset += (size_t) written;
        }
        if (::close(fd) != 0) ok = false;
        if (!ok) {
            (void) ::unlink(name.c_str());
            error = "microprobe write failed: " + name;
            return false;
        }
        return true;
    }

private:
    static void add32(std::vector<uint8_t>& out, uint32_t x) {
        for (unsigned i = 0; i < 4; ++i)
            out.push_back((uint8_t) (x >> (8 * i)));
    }
    static void add64(std::vector<uint8_t>& out, uint64_t x) {
        for (unsigned i = 0; i < 8; ++i)
            out.push_back((uint8_t) (x >> (8 * i)));
    }
    std::string prefix_;
    bool enabled_ = false;
};

} // namespace strata::heterogeneous
