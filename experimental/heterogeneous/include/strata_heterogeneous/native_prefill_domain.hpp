#pragma once
// Native chunk-major Prefill in each backend. Only the R-only transport is new.
// The caller owns weights/cache/session, and has completed its resident reset.
#include "strata_heterogeneous/native_prefill_init.hpp"
#include "strata_heterogeneous/native_capacity.hpp"
#include "strata_heterogeneous/prefill_stage_files.hpp"
#include "strata_heterogeneous/prefill_state_capture.hpp"
#include "strata_heterogeneous/boundary_capture.hpp"
#include "strata_heterogeneous/host.hpp"
#include "strata_heterogeneous/window_admission.hpp"
#include "strata_heterogeneous/mtp_request_policy.hpp"
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
    uint32_t long_context_chunk = 0;
};
inline bool run_native_prefill_domain(const Frame& cmd, int fd,
        const std::vector<NativePrefillStage>& stages, bool sender,
        int timeout_ms, std::string& error,
        prefill::Prefill::FirstTokenProbe first_probe = {},
        std::function<bool(const char*,std::string&)> entry_state = {},
        prefill::Prefill::FirstHcProbe hc_probe = {},
        std::function<bool(prefill::Prefill&,const float*,int64_t,int64_t,std::string&)> terminal_chunk = {}) {
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
        if (cmd.kind != Kind::prefill_spec || stages.size() != (sender ? 1u : 2u) ||
            cmd.stage_index < stages.size() - 1) {
            error = "native prefill domain command/topology mismatch"; return false;
        }
        const bool local = !sender && stages.front().session.layer_lo == 0;
        if (local && cmd.stage_index != 1) {
            error = "full native HIP pair requires terminal physical stage 1"; return false;
        }
        if ((first_probe || entry_state || hc_probe) && (!sender || !first_probe || !entry_state ||
            cmd.rows!=1 || cmd.prefill_chunk!=1 || cmd.request_id!=2)) {
            error="first-token prefill diagnostic requires one CUDA entry token";return false;
        }
        const char* native_capture_raw=std::getenv("STRATA_HETERO_NATIVE_STAGE_CAPTURE");
        const char* reference_raw=std::getenv("STRATA_HETERO_NATIVE_STAGE_REFERENCE");
        const std::string stage_capture=native_capture_raw?native_capture_raw:"";
        const std::string reference=reference_raw?reference_raw:"";
        const bool stage_probe=!stage_capture.empty();
        const bool isolated=!reference.empty();
        if ((local && (stage_probe || isolated)) || (isolated && !stage_probe) || (stage_probe &&
            (cmd.request_id!=2 || cmd.rows>prefill_stage_max_rows || cmd.prefill_chunk>cmd.rows ||
             cmd.geometry.hc!=4 || cmd.geometry.n_embd!=2560 || cmd.stage_index!=(sender?0u:2u) ||
             first_probe || entry_state || hc_probe ||
             !std::filesystem::path(stage_capture).is_absolute() ||
             (isolated && (!std::filesystem::path(reference).is_absolute() || reference==stage_capture))))) {
            error="invalid bounded native stage replay diagnostic";return false;
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
            if ((!s.long_context_chunk && cmd.rows > legacy_workload_context) ||
                (s.long_context_chunk && (cmd.prefill_chunk != s.long_context_chunk ||
                   !long_context_cells_valid(s.session.max_cells) || s.reserve_bytes < long_context_min_reserve))) {
                error = "native prefill exceeds negotiated context or chunk"; return false;
            }
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
                   : (lb != stages.front().geometry.n_layers ||
                      (!local && stages.front().session.layer_lo <= 0))) {
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
                !isolated && i+1<stages.size()?owned.nodes[i+1].pf.get():nullptr);
        auto& first=owned.nodes.front();
        if (!local && (!check(cudaSetDevice(first.device),"prefill bridge device") ||
            !check(cudaMallocHost(reinterpret_cast<void**>(&owned.pinned),size_t(chunk)*stride*sizeof(float)),
                   "prefill bridge pinned staging"))) return false;
        // Physical receiving stage, not the terminal HIP domain endpoint (2).
        const size_t boundary_stage = cmd.stage_index - (stages.size() - 1);
        const char* capture_prefix = std::getenv("STRATA_HETERO_PREFILL_CAPTURE_PREFIX");
        const bool capture_bridge = capture_prefix && *capture_prefix;
        if (local && capture_bridge) {
            error = "cross-runtime prefill capture is not valid on a full local domain"; return false;
        }
        auto verify_piece = [&](const float* device, size_t off, size_t count,
                                int64_t pos, const char* direction, cudaStream_t stream, std::string& err) {
            return capture_device_handoff(device + off * stride, owned.pinned + off * stride,
                count * stride * sizeof(float), boundary_stage, cmd.request_id,
                uint64_t(pos) + off + 1, direction, stream,
                [&](std::string& e) {
                    const auto rc = cudaStreamSynchronize(stream);
                    if (rc == cudaSuccess) return true;
                    e = std::string("native prefill capture fence: ") + cudaGetErrorString(rc);
                    return false;
                }, err, "STRATA_HETERO_PREFILL_CAPTURE_PREFIX");
        };
        // Capture actual stage R, independently of pinned bridge staging.
        // Each callback runs on the owning stage's native thread/stream.
        if (stage_probe) {
            for (size_t i=0;i<stages.size();++i) {
                const size_t stage=boundary_stage+i;
                const int device=owned.nodes[i].device;
                owned.nodes[i].pf->on_stage_residual=[&,stage,device](const char* direction,const float* data,
                    int64_t rows,int64_t width,int64_t pos,void* stream,std::string& err) {
                    try {
                        int current=-1;
                        if (!prefill_stage_shape(stage,pos,rows,size_t(width)) || pos+rows>n ||
                            !data || !stream || cudaGetDevice(&current)!=cudaSuccess || current!=device ||
                            cudaStreamSynchronize(static_cast<cudaStream_t>(stream))!=cudaSuccess) {
                            err="native stage R observation owner/shape/fence failed";return false;
                        }
                        std::vector<float> values(size_t(rows)*stride);
                        if (cudaMemcpy(values.data(),data,values.size()*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess) {
                            err="native stage R readback failed";return false;
                        }
                        return prefill_stage_write(prefill_stage_file(stage_capture,stage,pos,direction),values.data(),values.size(),err);
                    } catch(const std::exception& ex) {err=ex.what();return false;}
                };
            }
        }
        int64_t offset=0;
        auto shape = [&](int64_t rows,int64_t pos) {
            return pos==offset && offset<n && rows==std::min(chunk,n-offset);
        };
        if (sender) {
            first.pf->on_first_token=std::move(first_probe);
            first.pf->on_first_hc=std::move(hc_probe);
            first.pf->set_external_handoff({},true);
            first.pf->on_chunk=[&](const float* r,int64_t rows,int64_t pos,std::string& err) {
                try {
                    if (!shape(rows,pos) || !r) { err="native prefill output chunk mismatch"; return false; }
                    // Before even the first D2H/transport operation.
                    if (entry_state && !entry_state("before_bridge",err)) return false;
                    if (!check(cudaMemcpy(owned.pinned,r,size_t(rows)*stride*sizeof(float),cudaMemcpyDeviceToHost),
                               "native prefill residual D2H")) return false;
                    for (size_t off=0;off<size_t(rows);) {
                        const size_t count=std::min(cap,size_t(rows)-off);
                        if (capture_bridge && !verify_piece(r, off, count, pos, "out", first.stream, err)) return false;
                        Frame f;
                        f.kind=Kind::prefill_rows; f.stage_index=cmd.stage_index; f.request_id=cmd.request_id;
                        f.window_id=uint64_t(pos)+off+1; f.plan_fingerprint=cmd.plan_fingerprint;
                        f.rows=static_cast<uint32_t>(count); f.geometry=cmd.geometry;
                        f.payload.assign(owned.pinned+off*stride,owned.pinned+(off+count)*stride);
                        if (!activation_finite(reinterpret_cast<const uint8_t*>(f.payload.data()),f.payload.size()*sizeof(float))) {
                            err="native prefill nonfinite output"; return false;
                        }
                        if (!isolated) send_frame(fd,f,timeout_ms);
                        off+=count;
                    }
                    offset+=rows; return true;
                } catch (const std::exception& ex) { err=ex.what(); return false; }
            };
        } else if (!local) {
            first.pf->set_external_handoff([&](float* dst,int64_t rows,int64_t width,
                    int64_t pos,void* stream,std::string& err) {
                try {
                    if (!shape(rows,pos) || width!=static_cast<int64_t>(stride) || !dst || !stream) {
                        err="native prefill input chunk mismatch"; return false;
                    }
                    if (isolated) {
                        if (!prefill_stage_read(prefill_stage_file(reference,0,pos,"out"),owned.pinned,
                                                size_t(rows)*stride,err)) return false;
                    } else for (size_t off=0;off<size_t(rows);) {
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
                    for (size_t off = 0; capture_bridge && off < size_t(rows);) {
                        const size_t count = std::min(cap, size_t(rows) - off);
                        if (!verify_piece(dst, off, count, pos, "in", cs, err)) return false;
                        off += count;
                    }
                    offset+=rows; return true;
                } catch (const std::exception& ex) { err=ex.what(); return false; }
            },isolated); // an isolated HIP0 emits its own R, not a live successor
            if (isolated) {
                // HIP0 and HIP1 execute separately, with immutable producer R
                // from the completed wire run. Neither consumes this run's
                // earlier stage's output or uses the native local link.
                first.pf->on_chunk=[](const float*,int64_t,int64_t,std::string&){return true;};
                owned.nodes[1].pf->set_external_handoff([&](float* dst,int64_t rows,int64_t width,
                        int64_t pos,void* stream,std::string& err) {
                    try {
                        int current=-1;
                        if (!dst || !stream || !prefill_stage_shape(2,pos,rows,size_t(width)) || pos+rows>n ||
                            cudaGetDevice(&current)!=cudaSuccess || current!=owned.nodes[1].device) {
                            err="isolated HIP1 input owner/shape failed";return false;
                        }
                        std::vector<float> values(size_t(rows)*stride);
                        if (!prefill_stage_read(prefill_stage_file(reference,1,pos,"out"),values.data(),values.size(),err))
                            return false;
                        // Synchronous host transfer; no borrowed pointer survives.
                        if (cudaMemcpy(dst,values.data(),values.size()*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess ||
                            cudaDeviceSynchronize()!=cudaSuccess) {
                            err="isolated HIP1 input upload failed";return false;
                        }
                        return true;
                    } catch(const std::exception& ex) {err=ex.what();return false;}
                },false);
            }
        }
        const bool mtp_bulk=bool(terminal_chunk);
        if (terminal_chunk) {
            if (sender || stage_probe || isolated || first_probe || entry_state || hc_probe || capture_bridge) {
                error="terminal residual consumer requires ordinary grouped receiving prefill";return false;
            }
            // The final stage alone owns the post-layer residual. The native
            // callback synchronizes its producer stream and fences reuse until
            // the consumer returns. Never install a callback on the earlier GPU.
            auto* terminal=owned.nodes.back().pf.get();
            terminal->on_chunk=[terminal, consume=std::move(terminal_chunk)](
                    const float* rows,int64_t count,int64_t pos,std::string& err) {
                return consume(*terminal,rows,count,pos,err);
            };
        }
        // Initialize native owners on their actual devices. No cache loans or recreation.
        for (size_t i=0;i<stages.size();++i) {
            auto& node=owned.nodes[i]; const auto& s=stages[i];
            // Keep prime/graph headroom on the terminal GPU while target
            // prefill scratch is live. Ordinary/legacy prefill is unchanged.
            const uint64_t extra=mtp_bulk && i+1==stages.size()?MtpRequestPolicy::prime_headroom_bytes:0;
            if (extra>std::numeric_limits<uint64_t>::max()-s.reserve_bytes) {
                error="MTP prefill reserve overflow";return false;
            }
            const uint64_t reserve=s.reserve_bytes+extra;
            if ((s.long_context_chunk || mtp_bulk) && !check_native_prefill_capacity(node.device,boundary_stage+i,
                    "prefill_before",s.geometry,s.session,static_cast<uint32_t>(chunk),reserve,error)) return false;
            if (!check(cudaSetDevice(node.device),"prefill stage device") ||
                !init_owned_native_prefill(*node.pf,s.weights,s.geometry,s.session,s.experts,
                                           chunk,node.stream,reserve,0,error)) return false;
            if ((s.long_context_chunk || mtp_bulk) && !sample_native_capacity(node.device,boundary_stage+i,
                    "prefill_allocated",reserve,0,error)) return false;
        }
        if (entry_state && !entry_state("after_init",error)) return false;
        std::vector<int64_t> tokens(cmd.window->token_ids.begin(),cmd.window->token_ids.end());
        if (!check(cudaSetDevice(first.device),"prefill entry device") ||
            !first.pf->run_external(tokens.data(),n,0,chunk,n>chunk,error,local)) return false;
        if (!local && offset!=n) { error="native prefill boundary did not transfer entire prompt"; return false; }
        if (isolated && !sender) {
            if (!check(cudaSetDevice(first.device),"isolated HIP0 drain owner") || !first.pf->idle(error) ||
                !check(cudaSetDevice(owned.nodes[1].device),"isolated HIP1 owner") ||
                !owned.nodes[1].pf->run_external(tokens.data(),n,0,chunk,n>chunk,error)) return false;
        }
        for (size_t i=0;i<owned.nodes.size();++i) {
            auto& node = owned.nodes[i];
            if (!check(cudaSetDevice(node.device),"prefill drain device") || !node.pf->idle(error)) return false;
            if ((stages[i].long_context_chunk || mtp_bulk) && !sample_native_capacity(node.device,boundary_stage+i,
                    "prefill_drained",stages[i].reserve_bytes,0,error)) return false;
        }
        if (stage_probe) {
            for (size_t i=0;i<stages.size();++i) {
                PrefillStateCapture state;
                const auto& s=stages[i];
                if (!check(cudaSetDevice(owned.nodes[i].device),"native stage checkpoint owner") ||
                    !state.configure_native_stage(stage_capture,boundary_stage+i,n,s.session,s.geometry,error) ||
                    !state.observe(cmd.request_id,n,s.session,s.geometry,error)) return false;
            }
        }
        return true;
    } catch (const std::exception& ex) { error=ex.what(); return false; }
}
} // namespace strata::heterogeneous
