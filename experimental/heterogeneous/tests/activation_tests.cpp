// CPU-only validation of the exact R|bo|inject activation view and staged
// executor contract.  No GPU API is linked or initialized.
#include "strata_heterogeneous/host.hpp"
#include "strata_heterogeneous/shared.hpp"

#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace strata::heterogeneous;
namespace {
int assertions = 0;
void check(bool value, const char* message) {
    ++assertions;
    if (!value) throw std::runtime_error(message);
}
template <typename F> void fails(F&& fn, const char* message) {
    bool caught = false;
    try { fn(); } catch (const std::exception&) { caught = true; }
    check(caught, message);
}
Frame frame(const Plan& p, uint64_t request = 8, uint64_t window = 5, uint32_t rows = 2) {
    Frame result;
    result.request_id = request;
    result.window_id = window;
    result.plan_fingerprint = p.fingerprint();
    result.geometry = p.geometry;
    result.rows = rows;
    const size_t count = p.geometry.floats_per_row() * rows;
    result.payload.resize(count);
    for (size_t i = 0; i < count; ++i) result.payload[i] = static_cast<float>(i % 13) / 13.f;
    return result;
}
class OnlyVectorExecutor final : public StageExecutor {
public:
    bool prepare(const Plan& plan, size_t index, std::string&) override {
        stage_ = plan.stages.at(index);
        return true;
    }
    bool run_window(const Frame& in, Frame& out, std::string& error) override {
        ++vector_calls;
        return mock_execute(stage_, in, out, error);
    }
    bool commit_window(uint64_t, uint64_t, std::string&) override { return true; }
    void abandon_window() noexcept override {}
    void shutdown() noexcept override {}
    unsigned vector_calls = 0;
private:
    Stage stage_{};
};
class BrokenVectorExecutor final : public StageExecutor {
public:
    bool prepare(const Plan&, size_t, std::string&) override { return true; }
    bool run_window(const Frame& in, Frame& out, std::string&) override {
        out = in;
        out.payload.pop_back();
        return true;
    }
    bool commit_window(uint64_t, uint64_t, std::string&) override { return true; }
    void abandon_window() noexcept override {}
    void shutdown() noexcept override {}
};
class SharedOnlyExecutor final : public StageExecutor {
public:
    bool prepare(const Plan& plan, size_t index, std::string& error) override {
        return mock_.prepare(plan, index, error);
    }
    bool run_window(const Frame&, Frame&, std::string&) override {
        throw std::logic_error("vector run must not be called by shared worker");
    }
    bool run_staged_window(const Frame& in, ActivationView view, std::string& error) override {
        return mock_.run_staged_window(in, view, error);
    }
    bool commit_window(uint64_t r, uint64_t w, std::string& e) override {
        return mock_.commit_window(r, w, e);
    }
    void abandon_window() noexcept override { mock_.abandon_window(); }
    void shutdown() noexcept override { mock_.shutdown(); }
private:
    MockStageExecutor mock_;
};
class WriteAndFailExecutor final : public StageExecutor {
public:
    bool prepare(const Plan&, size_t, std::string&) override { return true; }
    bool run_window(const Frame&, Frame&, std::string&) override { throw std::logic_error("unexpected vector call"); }
    bool run_staged_window(const Frame&, ActivationView view, std::string& error) override {
        view.row(0).residual.write_f32(0, 99.f);
        error = "injected staged worker failure after mapping write";
        return false;
    }
    bool commit_window(uint64_t, uint64_t, std::string&) override { return false; }
    void abandon_window() noexcept override {}
    void shutdown() noexcept override {}
};
void test_layout_and_validation() {
    std::vector<uint8_t> buffer(2 * 11 * sizeof(float));
    ActivationView view(buffer.data(), buffer.size(), {2, 3}, 2);
    check(view.bytes() == 88 && view.rows() == 2, "view dimensions");
    const auto a = view.row(0), b = view.row(1);
    check(a.residual.bytes == 24 && a.pending_write.bytes == 12 && a.inject.bytes == 8,
          "R, bo, inject byte extents follow Strata verifier layout");
    check(a.pending_write.data == a.residual.data + 24 && a.inject.data == a.pending_write.data + 12,
          "row 0 arrays tightly packed in correct order");
    check(b.residual.data == a.residual.data + 44 &&
          b.inject.data + b.inject.bytes == buffer.data() + buffer.size(),
          "independent rows contiguous and no excess bytes");
    a.residual.write_f32(0, 1.25f);
    a.pending_write.write_f32(0, -3.f);
    b.inject.write_f32(1, 42.f);
    check(a.residual.read_f32(0) == 1.25f && a.pending_write.read_f32(0) == -3.f &&
          b.inject.read_f32(1) == 42.f, "byte accessors preserve float values");
    uint32_t bits = 0x7fc12345u;
    std::memcpy(a.inject.data, &bits, sizeof(bits));
    float nan_value = a.inject.read_f32(0);
    uint32_t observed = 0;
    std::memcpy(&observed, &nan_value, sizeof(observed));
    check(observed == bits, "raw NaN payload bits preserved");
    fails([&] { (void) view.row(2); }, "row bounds checked");
    fails([&] { (void) a.pending_write.read_f32(3); }, "read element bounds checked");
    fails([&] { b.inject.write_f32(2, 1.f); }, "write element bounds checked");
    fails([&] { (void) ActivationView(buffer.data(), buffer.size() - 1, {2, 3}, 2); },
          "exact buffer extent required");
    fails([&] { (void) ActivationView(buffer.data(), buffer.size(), {2, 3}, 0); },
          "zero rows rejected");
    fails([&] { (void) ActivationView(nullptr, buffer.size(), {2, 3}, 2); },
          "null buffer rejected");
    fails([&] { (void) ActivationView(buffer.data(), buffer.size(), {UINT32_MAX, UINT32_MAX}, 2); },
          "oversized geometry rejected before touching memory");
    fails([&] { (void) ActivationView(buffer.data(), buffer.size(), {2, 3}, UINT32_MAX); },
          "oversized row count rejected without overflow");
}
void test_vector_fallback_and_direct_mock() {
    const Plan p = parse_plan("cuda:0:0-4", 4, {2, 3}, 111);
    Frame in = frame(p);
    std::string error;
    OnlyVectorExecutor fallback;
    check(fallback.prepare(p, 0, error), "vector fallback prepares");
    auto bytes = std::vector<uint8_t>(in.payload.size() * sizeof(float));
    std::memcpy(bytes.data(), in.payload.data(), bytes.size());
    Frame metadata = in;
    metadata.payload.clear();
    ActivationView view(bytes.data(), bytes.size(), in.geometry, in.rows);
    check(fallback.run_staged_window(metadata, view, error), "default vector fallback succeeds");
    check(fallback.vector_calls == 1, "default fallback called existing vector API once");
    Frame malformed = metadata;
    malformed.rows = 1;
    fails([&] { (void) fallback.run_staged_window(malformed, view, error); },
          "default fallback rejects mismatched row metadata");
    check(fallback.vector_calls == 1, "bad metadata never executes backend");
    Frame expected;
    check(mock_execute(p.stages[0], in, expected, error), "reference mock execution");
    check(std::memcmp(bytes.data(), expected.payload.data(), bytes.size()) == 0,
          "vector fallback publishes exact reference bits");
    // The direct mock executor modifies only the mapped byte slot, without a
    // full-vector allocation or an extra data-frame socket copy.
    std::memcpy(bytes.data(), in.payload.data(), bytes.size());
    MockStageExecutor direct;
    check(direct.prepare(p, 0, error), "direct stage prepares");
    check(direct.run_staged_window(metadata, view, error), "direct byte-buffer execution succeeds");
    check(std::memcmp(bytes.data(), expected.payload.data(), bytes.size()) == 0,
          "direct byte-buffer arithmetic identical to vector path");
    check(!direct.run_staged_window(metadata, view, error), "second provisional window blocked");
    check(!direct.run_staged_window(malformed, view, error), "direct worker rejects a second invalid window");
    check(direct.commit_window(metadata.request_id, metadata.window_id, error), "direct staged commit succeeds");
    std::memcpy(bytes.data(), in.payload.data(), bytes.size());
    check(direct.run_staged_window(metadata, view, error), "direct stage reusable after commit");
    direct.abandon_window();
    direct.shutdown();
    // Any invalid default-fallback output must be rejected before the
    // shared mapping is modified (important for partially failed kernels).
    std::memcpy(bytes.data(), in.payload.data(), bytes.size());
    BrokenVectorExecutor broken;
    fails([&] { (void) broken.run_staged_window(metadata, view, error); },
          "invalid vector fallback shape rejected");
    check(std::memcmp(bytes.data(), in.payload.data(), bytes.size()) == 0,
          "invalid vector fallback left mapped slot untouched");
}
void test_worker_only_direct_and_fail_closed() {
    const Plan p = parse_plan("hip:0:0-4", 4, {2, 3}, 111);
    auto run_one = [&](bool injected_failure) {
        int fd[2]{};
        check(::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0, "test worker socketpair");
        const pid_t pid = ::fork();
        check(pid >= 0, "test worker fork");
        if (pid == 0) {
            ::close(fd[0]);
            try {
                if (injected_failure) {
                    WriteAndFailExecutor stage;
                    serve_worker_stateful(fd[1], p, 0, stage, 1000, true);
                } else {
                    SharedOnlyExecutor stage;
                    serve_worker_stateful(fd[1], p, 0, stage, 1000, true);
                }
                ::close(fd[1]);
                ::_exit(0);
            } catch (...) { ::_exit(2); }
        }
        ::close(fd[1]);
        PipelineSession session(p, {fd[0]}, 1000, true);
        session.initialize();
        Frame initial = frame(p, 9, 1);
        if (injected_failure) {
            std::string failure;
            try { (void) session.run_window(initial); }
            catch (const std::exception& ex) { failure = ex.what(); }
            check(failure.find("injected staged worker failure") != std::string::npos,
                  "post-write failure propagated to coordinator");
            check(session.poisoned(), "coordinator poisoned after worker mutated mapped slot");
            fails([&] { (void) session.run_window(frame(p, 9, 2)); },
                  "poisoned session never publishes later window");
        } else {
            auto result = session.run_window(initial);
            Frame expected;
            std::string error;
            check(mock_execute(p.stages[0], initial, expected, error), "direct worker reference");
            check(result.payload == expected.payload, "worker used direct staged hook, not vector API");
            check(!session.poisoned(), "successful stage remains usable");
        }
        ::close(fd[0]);
        int status = 0;
        check(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "worker stops without deadlock");
    };
    run_one(false);
    run_one(true);
}
} // namespace
int main() {
    try {
        test_layout_and_validation();
        test_vector_fallback_and_direct_mock();
        test_worker_only_direct_and_fail_closed();
        std::cout << "activation view tests: PASS (" << assertions
                  << " assertions, in-place and fallback staged workers, no GPU)\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "activation view tests: FAIL: " << ex.what() << '\n';
        return 1;
    }
}
