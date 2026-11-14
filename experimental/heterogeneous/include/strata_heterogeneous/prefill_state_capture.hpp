#pragma once

// CUDA entry worker only. Opt-in, read-only committed-prefix snapshots using
// vanilla checkpoint/KV capture and their owned-layer layouts, not new kernels.
// The observer is called after the serial commit fence or native prompt drain.
#include "strata/core/conversation_snapshot.hpp"
#include <cuda_runtime.h>

#include <array>
#include <cerrno>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

namespace strata::heterogeneous {

class PrefillStateCapture {
public:
    bool active() const noexcept { return !prefix_.empty(); }
    bool first_token() const noexcept { return first_token_; }

    // Observe the first GDN layer and PLE history. No inference writes.
    bool entry_state(uint64_t request,const char* phase,const core::SessionState& ss,
                     const core::ModelGeometry& g,std::string& error) const {
        if (!first_token_) return true;
        const bool reset=std::strcmp(phase,"reset")==0;
        if (request<1 || request>3 || (!reset && request!=2) ||
            (!reset && std::strcmp(phase,"after_init") && std::strcmp(phase,"before_bridge") &&
             std::strcmp(phase,"after_drain"))) {error="invalid first-token state phase";return false;}
        const size_t rec=size_t(g.ssm_state_size)*g.ssm_v_heads*g.ssm_state_size;
        const size_t conv=size_t(g.ssm_conv_channels)*(g.ssm_d_conv-1);
        const size_t ple=size_t(strata::kernels::NG_HIST)*strata::kernels::NG_HC_DIM;
        if (!ss.gdn_state || !ss.ple_hist || !rec || !conv || !ple) {error="missing first-token state";return false;}
        int current=-1;
        if (cudaGetDevice(&current)!=cudaSuccess || current!=device_ ||
            cudaDeviceSynchronize()!=cudaSuccess) {error="first-token state owner/drain failed";return false;}
        const std::string base=std::string(phase)+".";
        return sample(request,base+"recurrence",ss.gdn_state,rec,nullptr,error) &&
               sample(request,base+"conv",ss.gdn_state+rec,conv,nullptr,error) &&
               sample(request,base+"ple",ss.ple_hist,ple,nullptr,error);
    }
    bool native_sample(const char* name,const float* ptr,size_t count,void* stream,std::string& error) const {
        if (!first_token_ || (std::strcmp(name,"embedding") && std::strcmp(name,"hc_mixed") &&
            std::strcmp(name,"gdn_qkv")) || !stream) {error="invalid native first-token sample";return false;}
        return sample(2,name,ptr,count,stream,error);
    }

    bool configure(size_t stage, const core::SessionState& ss,
                   const core::ModelGeometry& g, std::string& error) {
        const char* prefix = std::getenv("STRATA_HETERO_PREFILL_STATE_PREFIX");
        const char* at = std::getenv("STRATA_HETERO_PREFILL_STATE_AT");
        if ((!prefix || !*prefix) && (!at || !*at)) return true;
        if (!prefix || !*prefix || !at || !*at || stage != 0 ||
            ss.layer_lo != 0 || ss.layer_hi <= 1 || ss.layer_hi > g.n_layers ||
            ss.gdn_alloc <= 0 || ss.qsa_alloc <= 0 || !ss.ple.ready() ||
            ss.gdn_ord0 < 0 || ss.qsa_ord0 < 0) {
            error = "state capture requires CUDA entry stage and its real PLE/GDN/QSA owners"; return false;
        }
        const char* end = at + std::strlen(at);
        const auto parsed = std::from_chars(at, end, cut_);
        const std::filesystem::path path(prefix);
        if (parsed.ec != std::errc{} || parsed.ptr != end || cut_ < 1 || cut_ > 16 ||
            cut_ >= ss.max_cells || !path.is_absolute() || path.filename().empty() ||
            std::strlen(prefix) > 4096 || !std::filesystem::is_directory(path.parent_path())) {
            error = "invalid state capture path or committed prefix extent"; return false;
        }
        // The vanilla checkpoint's optional async/gather modes allocate VRAM.
        // This diagnostic must not change the admitted resident memory budget.
        for (const char* flag : {"STRATA_CKPT_ASYNC", "STRATA_CKPT_GATHER"}) {
            const char* value = std::getenv(flag);
            if (value && value[0] == '1') {
                error = std::string("state capture requires synchronous vanilla snapshots: ") + flag;
                return false;
            }
        }
        prefix_ = prefix;
        const char* first=std::getenv("STRATA_HETERO_PREFILL_FIRST_TOKEN");
        first_token_=first && std::strcmp(first,"1")==0;
        if (first_token_ && cut_!=1) {error="first-token localization requires cut 1";return false;}
        if (cudaGetDevice(&device_) != cudaSuccess) {
            error = "state capture cannot identify CUDA owner"; return false;
        }
        return true;
    }

    bool observe(uint64_t request, int64_t position, const core::SessionState& ss,
                 const core::ModelGeometry& g, std::string& error) const {
        if (!active() || position != cut_) return true;
        if (request < 1 || request > 3) {
            error = "state capture request outside A/B/A"; return false;
        }
        int current = -1;
        if (cudaGetDevice(&current) != cudaSuccess || current != device_ ||
            cudaDeviceSynchronize() != cudaSuccess) {
            error = "state capture owner or drain failed"; return false;
        }
        core::ConversationStateSizes sizes;
        if (!core::conversation_session_sizes(g, ss, sizes, error)) return false;
        // Bound the diagnostic independently of the context size. No extra GPU
        // storage; the native snapshot path reads the authoritative K/V pools.
        constexpr size_t cap = 256u * 1024u * 1024u;
        size_t total = 0;
        auto budget = [&](uint64_t bytes) {
            if (bytes > cap - total) return false;
            total += static_cast<size_t>(bytes); return true;
        };
        if (!budget(sizes.gdn) || !budget(sizes.ple)) {
            error = "state capture exceeds 256 MiB limit"; return false;
        }
        for (int64_t j = 0; j < ss.qsa_alloc; ++j) {
            std::array<uint64_t, 5> part{};
            if (!core::conversation_kv_part_sizes(ss.qsa_states[ss.qsa_ord0+j], g, position, true, part, error))
                return false;
            if (!budget(sizes.tail) || !budget(sizes.dead) || !budget(sizes.block_pos)) {
                error = "state capture indexer exceeds limit"; return false;
            }
            for (uint64_t n : part)
                if (!budget(n)) { error = "state capture K/V exceeds limit"; return false; }
        }
        core::ConversationCheckpoint checkpoint;
        // The native capture validates the extent using ids.size(). No dummy
        // token IDs are serialized; the coordinator records the actual sequence.
        checkpoint.ids.resize(static_cast<size_t>(position));
        if (!core::conversation_checkpoint_save(checkpoint, ss, g, error)) return false;
        core::conversation_checkpoint_join(checkpoint.gdn.data());
        if (checkpoint.gdn.size() != sizes.gdn || checkpoint.ple.size() != sizes.ple ||
            checkpoint.tails.size() != size_t(ss.qsa_alloc) * sizes.tail ||
            checkpoint.dead.size() != size_t(ss.qsa_alloc) * sizes.dead ||
            checkpoint.block_pos.size() != size_t(ss.qsa_alloc) * sizes.block_pos ||
            sizes.gdn % size_t(ss.gdn_alloc) != 0) {
            error = "state capture native checkpoint shape mismatch"; return false;
        }
        const std::string stem = prefix_ + ".state.r" + std::to_string(request) +
                                 ".p" + std::to_string(position);
        std::ostringstream fields, qsa;
        bool first = true;
        auto emit = [&](const std::string& name, const char* type, size_t count, auto&& writer) {
            if (!write_file(stem + "." + name + ".bin", writer, error)) return false;
            if (!first) fields << ',';
            first = false;
            fields << "{\"name\":\"" << name << "\",\"type\":\"" << type
                   << "\",\"bytes\":" << count << '}';
            return true;
        };
        auto raw = [&](const std::string& name, const char* type, const uint8_t* data, size_t count) {
            return emit(name, type, count, [&](int fd) { return write_all(fd, data, count, error); });
        };
        if (!raw("ple.history", "f32", checkpoint.ple.data(), sizes.ple)) return false;
        const size_t gdn_row = sizes.gdn / size_t(ss.gdn_alloc);
        const size_t recurrent = size_t(g.ssm_state_size) * size_t(g.ssm_v_heads) *
                                 size_t(g.ssm_state_size) * sizeof(float);
        if (!recurrent || recurrent >= gdn_row) {
            error = "state capture GDN recurrence shape mismatch"; return false;
        }
        for (int64_t j = 0; j < ss.gdn_alloc; ++j) {
            const std::string name = "gdn" + std::to_string(ss.gdn_ord0 + j);
            const auto* row = checkpoint.gdn.data() + size_t(j) * gdn_row;
            if (!raw(name + ".recurrence", "f32", row, recurrent) ||
                !raw(name + ".conv", "f32", row + recurrent, gdn_row - recurrent)) return false;
        }
        for (int64_t j = 0; j < ss.qsa_alloc; ++j) {
            const int64_t ordinal = ss.qsa_ord0 + j;
            const std::string name = "qsa" + std::to_string(ordinal);
            if (!raw(name + ".tail", "f32", checkpoint.tails.data()+size_t(j)*sizes.tail, sizes.tail) ||
                !raw(name + ".dead", "f32", checkpoint.dead.data()+size_t(j)*sizes.dead, sizes.dead) ||
                !raw(name + ".block_pos", "i32", checkpoint.block_pos.data()+size_t(j)*sizes.block_pos,
                     sizes.block_pos)) return false;
            core::ConversationKv kv;
            if (!core::conversation_kv_save(kv, ss.qsa_states[ordinal], g, position, true, error)) return false;
            if (kv.cells < position || kv.cells <= 0 || kv.page_size <= 0 || kv.pooled_rows <= 0) {
                error = "state capture invalid native K/V layout"; return false;
            }
            if (j) qsa << ',';
            qsa << "{\"ordinal\":" << ordinal << ",\"format\":" << kv.format
                << ",\"cells\":" << kv.cells << ",\"page_size\":" << kv.page_size
                << ",\"heads\":" << kv.heads << ",\"head_dim\":" << kv.head_dim
                << ",\"pooled_rows\":" << kv.pooled_rows << ",\"idx_dim\":" << kv.idx_dim << '}';
            auto kv_part = [&](const char* suffix, const core::ConversationBuffer& part, bool pooled) {
                // Native snapshots include entire pages. Exclude unwritten
                // tail cells from all K/V (and scale) comparisons. The indexer
                // pooled image includes the moving spare row by native contract.
                const size_t cells = static_cast<size_t>(kv.cells);
                if (!pooled && part.size() % cells) {
                    error = "state capture native K/V cell stride mismatch"; return false;
                }
                const size_t count = pooled ? part.size() : part.size()/cells*size_t(position);
                return emit(name + "." + suffix, pooled ? "f32" : "opaque", count, [&](int fd) {
                    return part.visit(0, count, [&](const uint8_t* data, size_t n, size_t) {
                        return write_all(fd, data, n, error);
                    });
                });
            };
            if (!kv_part("k", kv.k, false) || !kv_part("v", kv.v, false) ||
                !kv_part("k_scale", kv.k_scale, false) || !kv_part("v_scale", kv.v_scale, false) ||
                !kv_part("pooled", kv.pooled, true)) return false;
        }
        // Only a completed snapshot gets a manifest. Raw files alone never
        // establish a valid observation. ple_token is scratch, not committed history.
        std::ostringstream meta;
        meta << "{\"schema\":\"strata-heterogeneous-prefix-state/v1\",\"stage\":0"
             << ",\"request\":" << request << ",\"position\":" << position
             << ",\"layer_lo\":" << ss.layer_lo << ",\"layer_hi\":" << ss.layer_hi
             << ",\"gdn_ord0\":" << ss.gdn_ord0 << ",\"gdn_alloc\":" << ss.gdn_alloc
             << ",\"qsa_ord0\":" << ss.qsa_ord0 << ",\"qsa_alloc\":" << ss.qsa_alloc
             << ",\"ple_prev\":[" << ss.ple_prev[0] << ',' << ss.ple_prev[1] << ']'
             << ",\"qsa\":[" << qsa.str() << "],\"fields\":[" << fields.str() << "]}\n";
        const std::string json = meta.str();
        return write_file(stem + ".json", [&](int fd) {
            return write_all(fd, json.data(), json.size(), error);
        }, error);
    }

private:
    std::string prefix_;
    bool first_token_=false;
    bool sample(uint64_t request,const std::string& name,const float* ptr,size_t count,
                void* stream,std::string& error) const {
        int device=-1;
        if (!ptr || !count || count>(16u<<20)/sizeof(float) ||
            cudaGetDevice(&device)!=cudaSuccess || device!=device_) {
            error="first-token sample size/device refused";return false;
        }
        if (stream && cudaStreamSynchronize(static_cast<cudaStream_t>(stream))!=cudaSuccess) {
            error="first-token native compute stream failed";return false;
        }
        std::vector<float> data(count);
        if (cudaMemcpy(data.data(),ptr,count*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess) {
            error="first-token sample download failed";return false;
        }
        return write_file(prefix_+".first.r"+std::to_string(request)+"."+name+".f32",
                          [&](int fd){return write_all(fd,data.data(),count*sizeof(float),error);},error);
    }
    int64_t cut_ = 0;
    int device_ = -1;

    static bool write_all(int fd, const void* data, size_t bytes, std::string& error) {
        const auto* p = static_cast<const uint8_t*>(data);
        for (size_t at = 0; at < bytes;) {
            const ssize_t n = ::write(fd, p + at, bytes - at);
            if (n > 0) at += static_cast<size_t>(n);
            else if (n < 0 && errno == EINTR) continue;
            else { error = "state capture file write failed"; return false; }
        }
        return true;
    }
    template<class Writer>
    static bool write_file(const std::string& path, Writer&& writer, std::string& error) {
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) { error = std::string("state capture exclusive open: ") + std::strerror(errno); return false; }
        bool ok = false;
        try { ok = writer(fd); }
        catch (...) { (void)::close(fd); (void)::unlink(path.c_str()); throw; }
        if (::close(fd) != 0) { error = "state capture file close failed"; ok = false; }
        if (!ok) (void)::unlink(path.c_str());
        return ok;
    }
};

} // namespace strata::heterogeneous
