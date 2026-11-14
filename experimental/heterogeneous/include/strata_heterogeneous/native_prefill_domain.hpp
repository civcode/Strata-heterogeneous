#pragma once
// Native chunk-major Prefill in each backend. Only the R-only transport is new.
// The caller owns weights/cache/session, and has completed its resident reset.
#include "strata_heterogeneous/native_prefill_init.hpp"
#include "strata_heterogeneous/host.hpp"
#include "strata_heterogeneous/window_admission.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

namespace strata::heterogeneous {
struct NativePrefillStage {
    const core::WeightTable& weights;
    const core::ModelGeometry& geometry;
    core::SessionState& session;
    NativePrefillExpertView experts;
    uint64_t reserve_bytes;
    uint64_t arena_budget_bytes;
};
inline bool run_native_prefill_domain(const Frame& cmd, int fd,
        const std::vector<NativePrefillStage>& stages, bool sender,
        int timeout_ms, std::string& error) {
    struct Local {
        int device = -1;
        cudaStream_t stream = nullptr;
        std::unique_ptr<prefill::Prefill> pf;
    };
    struct Owned {
        std::vector<Local> nodes;
        float* pinned = nullptr;
        ~Owned() {
            // Settle every device before freeing anything another stage may use.
            for (auto& n : nodes)
                if (n.device >= 0 &&
                    (cudaSetDevice(n.device) != cudaSuccess ||
                     cudaDeviceSynchronize() != cudaSuccess)) std::_Exit(70);
            for (auto& n : nodes) {
                if (n.device < 0) continue;
                if (cudaSetDevice(n.device) != cudaSuccess) std::_Exit(70);
                n.pf.reset();
                if (n.stream && cudaStreamDestroy(n.stream) != cudaSuccess) std::_Exit(70);
            }
            if (pinned && (cudaSetDevice(nodes.front().device) != cudaSuccess ||
                           cudaFreeHost(pinned) != cudaSuccess)) std::_Exit(70);
        }
    } owned;
    auto check = [&](cudaError_t c, const char* where) {
        if (c == cudaSuccess) return true;
        error = std::string(where) + ": " + cudaGetErrorString(c);
        return false;
    };
    try {
        validate_wire_frame(cmd);
        if (cmd.kind != Kind::prefill_spec || stages.size() != (sender ? 1u : 2u)) {
            error = "native prefill domain command/topology mismatch"; return false;
        }
        const int64_t n = cmd.rows;
        const int64_t chunk = std::min<int64_t>(n,cmd.prefill_chunk);
        const size_t stride = static_cast<size_t>(cmd.geometry.hc)*cmd.geometry.n_embd;
        const size_t cap = prefill_piece_rows(cmd.geometry);
        if (!stride || stride > std::numeric_limits<size_t>::max()/sizeof(float)/chunk) {
            error = "native prefill staging size overflow"; return false;
        }
        int64_t lb = stages.front().session.layer_lo;
        for (const auto& s : stages) {
            if (!s.experts || s.arena_budget_bytes || s.session.layer_lo != lb ||
                s.session.layer_hi <= lb || s.session.max_cells <= n ||
                s.geometry.hc != cmd.geometry.hc || s.geometry.n_embd != cmd.geometry.n_embd ||
                s.geometry.n_layers != stages.front().geometry.n_layers) {
                error = "native prefill requires contiguous fully resident stages and owned memory; explicit arena budget not yet supported";
                return false;
            }
            lb = s.session.layer_hi;
        }
        if (sender ? (stages.front().session.layer_lo != 0 || lb >= stages.front().geometry.n_layers)
                   : (stages.front().session.layer_lo <= 0 || lb != stages.front().geometry.n_layers)) {
            error = "native prefill backend stage range mismatch"; return false;
        }
        owned.nodes.resize(stages.size());
        for (size_t i=0;i<stages.size();++i) {
            auto& node=owned.nodes[i];
            if (!check(cudaSetDevice(stages[i].experts.device),"prefill set device")) return false;
            node.device=stages[i].experts.device;
            if (!check(cudaStreamCreateWithFlags(&node.stream,cudaStreamNonBlocking),"prefill stream")) return false;
            node.pf=std::make_unique<prefill::Prefill>();
        }
        for (size_t i=0;i<stages.size();++i)
            owned.nodes[i].pf->set_stage(stages[i].session.layer_lo,stages[i].session.layer_hi,
                i+1<stages.size()?owned.nodes[i+1].pf.get():nullptr);
        auto& first=owned.nodes.front();
        if (!check(cudaSetDevice(first.device),"prefill bridge device") ||
            !check(cudaMallocHost(reinterpret_cast<void**>(&owned.pinned),size_t(chunk)*stride*sizeof(float)),
                   "prefill bridge pinned staging")) return false;
        int64_t offset=0;
        auto shape = [&](int64_t rows,int64_t pos) {
            return pos==offset && offset<n && rows==std::min(chunk,n-offset);
        };
        if (sender) {
            first.pf->set_external_handoff({},true);
            first.pf->on_chunk=[&](const float* r,int64_t rows,int64_t pos,std::string& err) {
                try {
                    if (!shape(rows,pos) || !r) { err="native prefill output chunk mismatch"; return false; }
                    if (!check(cudaMemcpy(owned.pinned,r,size_t(rows)*stride*sizeof(float),cudaMemcpyDeviceToHost),
                               "native prefill residual D2H")) return false;
                    for (size_t off=0;off<size_t(rows);) {
                        const size_t count=std::min(cap,size_t(rows)-off);
                        Frame f;
                        f.kind=Kind::prefill_rows; f.stage_index=cmd.stage_index; f.request_id=cmd.request_id;
                        f.window_id=uint64_t(pos)+off+1; f.plan_fingerprint=cmd.plan_fingerprint;
                        f.rows=static_cast<uint32_t>(count); f.geometry=cmd.geometry;
                        f.payload.assign(owned.pinned+off*stride,owned.pinned+(off+count)*stride);
                        if (!activation_finite(reinterpret_cast<const uint8_t*>(f.payload.data()),f.payload.size()*sizeof(float))) {
                            err="native prefill nonfinite output"; return false;
                        }
                        send_frame(fd,f,timeout_ms); off+=count;
                    }
                    offset+=rows; return true;
                } catch (const std::exception& ex) { err=ex.what(); return false; }
            };
        } else {
            first.pf->set_external_handoff([&](float* dst,int64_t rows,int64_t width,
                    int64_t pos,void* stream,std::string& err) {
                try {
                    if (!shape(rows,pos) || width!=static_cast<int64_t>(stride) || !dst || !stream) {
                        err="native prefill input chunk mismatch"; return false;
                    }
                    for (size_t off=0;off<size_t(rows);) {
                        const size_t count=std::min(cap,size_t(rows)-off);
                        auto f=receive_frame(fd,timeout_ms);
                        if (!f || f->kind!=Kind::prefill_rows || f->stage_index!=cmd.stage_index ||
                            f->request_id!=cmd.request_id || f->plan_fingerprint!=cmd.plan_fingerprint ||
                            f->window_id!=uint64_t(pos)+off+1 || f->rows!=count ||
                            f->geometry.hc!=cmd.geometry.hc || f->geometry.n_embd!=cmd.geometry.n_embd ||
                            f->payload.size()!=count*stride ||
                            !activation_finite(reinterpret_cast<const uint8_t*>(f->payload.data()),f->payload.size()*sizeof(float))) {
                            err="native prefill residual identity or finite gate failed"; return false;
                        }
                        std::memcpy(owned.pinned+off*stride,f->payload.data(),count*stride*sizeof(float));
                        off+=count;
                    }
                    auto cs=static_cast<cudaStream_t>(stream);
                    if (!check(cudaMemcpyAsync(dst,owned.pinned,size_t(rows)*stride*sizeof(float),
                                               cudaMemcpyHostToDevice,cs),"prefill residual H2D") ||
                        !check(cudaStreamSynchronize(cs),"prefill staging lifetime fence")) return false;
                    offset+=rows; return true;
                } catch (const std::exception& ex) { err=ex.what(); return false; }
            },false);
        }
        // Initialize native owners on their actual devices. No cache loans or recreation.
        for (size_t i=0;i<stages.size();++i) {
            auto& node=owned.nodes[i]; const auto& s=stages[i];
            if (!check(cudaSetDevice(node.device),"prefill stage device") ||
                !init_owned_native_prefill(*node.pf,s.weights,s.geometry,s.session,s.experts,
                                           chunk,node.stream,s.reserve_bytes,0,error)) return false;
        }
        std::vector<int64_t> tokens(cmd.window->token_ids.begin(),cmd.window->token_ids.end());
        if (!check(cudaSetDevice(first.device),"prefill entry device") ||
            !first.pf->run_external(tokens.data(),n,0,chunk,n>chunk,error)) return false;
        if (offset!=n) { error="native prefill boundary did not transfer entire prompt"; return false; }
        for (auto& node:owned.nodes)
            if (!check(cudaSetDevice(node.device),"prefill drain device") || !node.pf->idle(error)) return false;
        return true;
    } catch (const std::exception& ex) { error=ex.what(); return false; }
}
} // namespace strata::heterogeneous
