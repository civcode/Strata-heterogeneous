#include "strata_heterogeneous/device.hpp"

#include <cstring>
#include <exception>
#include <utility>

namespace strata::heterogeneous {
namespace {
// Operations report a meaningful phase even when the backend returns false
// with an empty diagnostic. Unexpected throws are also converted to failures
// so no success/commit acknowledgement escapes after a GPU runtime exception.
template <typename F>
bool checked_call(F&& action, const char* phase, std::string& error) {
    error.clear();
    try {
        if (action(error)) return true;
        if (error.empty()) error = std::string(phase) + " failed";
    } catch (const std::exception& e) {
        error = std::string(phase) + ": " + e.what();
    } catch (...) {
        error = std::string(phase) + ": unknown exception";
    }
    return false;
}
} // namespace

void DeviceStageExecutor::abort_once() noexcept {
    if (!aborted_) {
        aborted_ = true;
        driver_.abort();
    }
}
bool DeviceStageExecutor::fail(const char* phase, std::string& error) noexcept {
    poisoned_ = true;
    pending_ = false;
    has_spec_ = false;
    // Always stop pending device work before touching potentially allocating
    // diagnostics: this function must remain noexcept during failure cleanup.
    abort_once();
    if (error.empty()) {
        try { error = std::string(phase) + " failed"; }
        catch (...) {} // no possible exception may skip GPU abort
    }
    return false;
}

bool DeviceStageExecutor::prepare(const Plan& plan, size_t index, std::string& error) {
    if (shutdown_ || prepared_ || poisoned_) {
        error = "device executor already prepared, shut down or poisoned";
        return false;
    }
    try {
        plan.validate();
        if (index >= plan.stages.size()) {
            error = "invalid device stage index";
            return fail("prepare", error);
        }
        plan_ = plan;
        stage_index_ = index;
        if (!checked_call([&](std::string& e) { return driver_.prepare(plan_, stage_index_, e); },
                          "prepare", error))
            return fail("prepare", error);
        prepared_ = true;
        return true;
    } catch (const std::exception& ex) {
        error = std::string("prepare: ") + ex.what();
        return fail("prepare", error);
    }
}

bool DeviceStageExecutor::reset_request(uint64_t request, std::string& error) {
    if (!prepared_ || pending_ || has_spec_ || poisoned_ || shutdown_ || !request)
        return fail("resident reset validation", error);
    if (!checked_call([&](std::string& e) { return driver_.reset_request(request, e); },
                      "resident reset", error))
        return fail("resident reset", error);
    request_id_ = request;
    window_id_ = 0;
    spec_ = {};
    return true;
}

bool DeviceStageExecutor::begin_window(const WindowSpec& spec, uint64_t request_id,
                                       uint64_t window_id, std::string& error) {
    if (!prepared_ || pending_ || poisoned_ || shutdown_ || has_spec_ ||
        !request_id || !window_id) {
        error = "device cannot begin overlapping or invalid verifier window";
        return fail("begin verifier window", error);
    }
    try { spec.validate(static_cast<uint32_t>(spec.token_ids.size())); }
    catch (const std::exception& ex) {
        error = std::string("invalid verifier tokens: ") + ex.what();
        return fail("begin verifier window", error);
    }
    if (!checked_call([&](std::string& e) {
            return driver_.begin_window(spec, request_id, window_id, e);
        }, "begin verifier window", error))
        return fail("begin verifier window", error);
    spec_ = spec;
    has_spec_ = true;
    request_id_ = request_id;
    window_id_ = window_id;
    return true;
}

bool DeviceStageExecutor::run_staged_window(const Frame& metadata, ActivationView view,
                                             std::string& error) {
    if (!prepared_ || pending_ || poisoned_ || shutdown_) {
        error = "device stage unavailable or has uncommitted window";
        return false;
    }
    // Reject malformed inputs before scheduling any GPU work or borrowing a
    // pointer. The stage/plan and exact view geometry are not negotiable.
    if (metadata.kind != Kind::data || metadata.stage_index != stage_index_ ||
        metadata.plan_fingerprint != plan_.fingerprint() ||
        !metadata.request_id || !metadata.window_id || !metadata.payload.empty() ||
        metadata.shared_checksum || !metadata.error.empty() ||
        metadata.identity != std::array<uint32_t, 4>{} ||
        metadata.rows != view.rows() ||
        metadata.geometry.hc != plan_.geometry.hc ||
        metadata.geometry.n_embd != plan_.geometry.n_embd ||
        view.geometry().hc != plan_.geometry.hc ||
        view.geometry().n_embd != plan_.geometry.n_embd ||
        (has_spec_ && (metadata.request_id != request_id_ ||
                        metadata.window_id != window_id_ ||
                        metadata.rows != spec_.token_ids.size()))) {
        error = "invalid device window metadata or activation view";
        return fail("validate window", error);
    }
    const size_t bytes = view.bytes();
    if (!checked_call([&](std::string& e) { return driver_.enqueue_upload(view.data(), bytes, e); },
                      "enqueue upload", error) ||
        !checked_call([&](std::string& e) { return driver_.wait_upload(e); },
                      "wait upload", error) ||
        !checked_call([&](std::string& e) { return driver_.enqueue_execute(metadata, e); },
                      "enqueue execute", error) ||
        !checked_call([&](std::string& e) { return driver_.wait_execute(e); },
                      "wait execute", error) ||
        !checked_call([&](std::string& e) { return driver_.enqueue_download(view.data(), bytes, e); },
                      "enqueue download", error) ||
        !checked_call([&](std::string& e) { return driver_.wait_download(e); },
                      "wait download", error))
        return fail("staged device transfer", error);
    request_id_ = metadata.request_id;
    window_id_ = metadata.window_id;
    pending_ = true;
    return true;
}

bool DeviceStageExecutor::run_window(const Frame& input, Frame& output, std::string& error) {
    if (!prepared_ || pending_ || poisoned_ || shutdown_) {
        error = "device stage unavailable or has uncommitted window";
        return false;
    }
    try {
        // Existing socket transport takes a validated copy, then uses the
        // same DeviceDriver synchronization sequence as shared memory.
        validate_wire_frame(input);
        if (input.kind != Kind::data)
            throw std::invalid_argument("socket stage requires data frame");
        output = input;
        ActivationView view(reinterpret_cast<uint8_t*>(output.payload.data()),
                            output.payload.size() * sizeof(float), output.geometry, output.rows);
        // Data frames have no additional control metadata after validation.
        Frame metadata;
        metadata.stage_index = input.stage_index;
        metadata.request_id = input.request_id;
        metadata.window_id = input.window_id;
        metadata.plan_fingerprint = input.plan_fingerprint;
        metadata.rows = input.rows;
        metadata.geometry = input.geometry;
        if (!run_staged_window(metadata, view, error)) {
            output = Frame{};
            return false;
        }
        return true;
    } catch (const std::exception& ex) {
        error = std::string("socket staged window: ") + ex.what();
        output = Frame{};
        return fail("socket staged window", error);
    }
}

bool DeviceStageExecutor::commit_prefix(uint64_t request_id, uint64_t window_id,
                                        uint32_t n_keep, std::string& error) {
    if (!has_spec_ || !prepared_ || poisoned_ || shutdown_ || !pending_ ||
        request_id != request_id_ || window_id != window_id_ ||
        n_keep > spec_.token_ids.size()) {
        error = "device has no matching verifier prefix to commit";
        return fail("prefix commit validation", error);
    }
    if (!checked_call([&](std::string& e) {
            return driver_.enqueue_commit_prefix(request_id, window_id, n_keep, e);
        }, "enqueue prefix commit", error) ||
        !checked_call([&](std::string& e) { return driver_.wait_commit(e); },
                      "wait prefix commit", error))
        return fail("accepted-prefix commit", error);
    pending_ = false;
    has_spec_ = false;
    return true;
}

bool DeviceStageExecutor::commit_window(uint64_t request_id, uint64_t window_id,
                                         std::string& error) {
    if (!prepared_ || poisoned_ || shutdown_ || !pending_ || has_spec_ ||
        request_id != request_id_ || window_id != window_id_) {
        error = "device stage has no matching provisional window to commit";
        return fail("commit validation", error);
    }
    if (!checked_call([&](std::string& e) { return driver_.enqueue_commit(request_id, window_id, e); },
                      "enqueue commit", error) ||
        !checked_call([&](std::string& e) { return driver_.wait_commit(e); },
                      "wait commit", error))
        return fail("commit", error);
    pending_ = false;
    return true;
}
void DeviceStageExecutor::abandon_window() noexcept {
    // The worker/whole pipeline must be rebuilt after any interrupted GPU
    // operation: explicit abort does not promise a reliable state restore.
    poisoned_ = true;
    pending_ = false;
    has_spec_ = false;
    abort_once();
}
void DeviceStageExecutor::shutdown() noexcept {
    if (shutdown_) return;
    shutdown_ = true;
    if (pending_ || poisoned_) abort_once();
    pending_ = false;
    has_spec_ = false;
    driver_.shutdown();
    prepared_ = false;
}
} // namespace strata::heterogeneous
