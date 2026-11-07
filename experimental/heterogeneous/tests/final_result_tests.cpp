#include "strata_heterogeneous/final_result.hpp"

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
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
void ok(bool pass, const char* message) {
    ++assertions;
    if (!pass) throw std::runtime_error(message);
}
template<class F> void rejects(F&& fn, const char* why) {
    bool thrown = false;
    try { fn(); } catch (const std::exception&) { thrown = true; }
    ok(thrown, why);
}
FinalStageResult sample(bool logits = true) {
    FinalStageResult r;
    r.stage_index = 2;
    r.request_id = 120;
    r.window_id = 8;
    r.plan_fingerprint = 0x1144eeaa;
    r.rows = 3;
    r.vocab_size = 5;
    r.predicted_tokens = {1, 4, 0};
    r.nonfinite_rows = {1};
    if (logits) {
        r.logits = std::vector<float>(r.rows * r.vocab_size, 0.25f);
        uint32_t bits = 0x7fc0abcd; // preserve raw NaN diagnostics
        std::memcpy(&r.logits->at(0), &bits, sizeof(bits));
        r.logits->at(1) = -0.0f;
    }
    return r;
}
void raw_equal(const FinalStageResult& a, const FinalStageResult& b) {
    ok(a.stage_index == b.stage_index && a.request_id == b.request_id &&
       a.window_id == b.window_id && a.plan_fingerprint == b.plan_fingerprint &&
       a.rows == b.rows && a.vocab_size == b.vocab_size, "result identity preserved");
    ok(a.predicted_tokens == b.predicted_tokens && a.nonfinite_rows == b.nonfinite_rows,
       "token picks and nonfinite row markers preserved");
    ok(a.logits.has_value() == b.logits.has_value(), "optional logits presence preserved");
    if (a.logits) ok(a.logits->size() == b.logits->size() &&
                      std::memcmp(a.logits->data(), b.logits->data(),
                                  a.logits->size() * sizeof(float)) == 0,
                      "logits preserve exact bits including NaNs and signed zero");
}
void encoding_tests() {
    const auto r = sample();
    auto encoded = encode_final_result(r);
    ok(encoded.size() == final_result_header_bytes + 4 * (3 + 1 + 1 + 3 * 5),
       "header and body sizes match declared schema");
    raw_equal(r, decode_final_result(encoded));
    const auto no_logits = sample(false);
    raw_equal(no_logits, decode_final_result(encode_final_result(no_logits)));
    auto invalid = r;
    invalid.request_id = 0;
    rejects([&] { (void)encode_final_result(invalid); }, "missing request ID rejected");
    invalid = r; invalid.plan_fingerprint = 0;
    rejects([&] { (void)encode_final_result(invalid); }, "missing plan fingerprint rejected");
    invalid = r; invalid.window_id = 0;
    rejects([&] { (void)encode_final_result(invalid); }, "missing window ID rejected");
    invalid = r; invalid.rows = 0;
    rejects([&] { (void)encode_final_result(invalid); }, "zero rows rejected");
    invalid = r; invalid.rows = final_result_max_rows + 1;
    rejects([&] { (void)encode_final_result(invalid); }, "oversized row count rejected");
    invalid = r; invalid.vocab_size = 0;
    rejects([&] { (void)encode_final_result(invalid); }, "zero vocabulary rejected");
    invalid = r; invalid.vocab_size = UINT32_MAX;
    rejects([&] { (void)encode_final_result(invalid); }, "invalid vocabulary limit rejected");
    invalid = r; invalid.predicted_tokens.pop_back();
    rejects([&] { (void)encode_final_result(invalid); }, "missing prediction row rejected");
    invalid = r; invalid.predicted_tokens[0] = -1;
    rejects([&] { (void)encode_final_result(invalid); }, "negative prediction rejected");
    invalid = r; invalid.predicted_tokens[0] = 5;
    rejects([&] { (void)encode_final_result(invalid); }, "out-of-vocabulary prediction rejected");
    invalid = r; invalid.nonfinite_rows = {1, 1};
    rejects([&] { (void)encode_final_result(invalid); }, "duplicate nonfinite markers rejected");
    invalid = r; invalid.nonfinite_rows = {2, 1};
    rejects([&] { (void)encode_final_result(invalid); }, "unsorted nonfinite markers rejected");
    invalid = r; invalid.nonfinite_rows = {3};
    rejects([&] { (void)encode_final_result(invalid); }, "out-of-range nonfinite marker rejected");
    invalid = r; invalid.logits->pop_back();
    rejects([&] { (void)encode_final_result(invalid); }, "incorrect logits shape rejected");
    invalid = r; invalid.rows = 8192; invalid.vocab_size = 1024;
    invalid.predicted_tokens.assign(invalid.rows, 0);
    invalid.logits->assign(uint64_t(invalid.rows) * invalid.vocab_size, 0.0f);
    invalid.nonfinite_rows.clear();
    rejects([&] { (void)encode_final_result(invalid); }, "oversized logits payload refused");
    require_final_result_identity(r, 2, 120, 8, r.plan_fingerprint, 3);
    rejects([&] { require_final_result_identity(r, 1, 120, 8, r.plan_fingerprint, 3); }, "wrong stage rejected");
    rejects([&] { require_final_result_identity(r, 2, 120, 9, r.plan_fingerprint, 3); }, "wrong window rejected");
    rejects([&] { require_final_result_identity(r, 2, 120, 8, 111, 3); }, "wrong plan rejected");
    rejects([&] { require_final_result_identity(r, 2, 120, 8, r.plan_fingerprint, 2); }, "wrong rows rejected");

    for (size_t pos : {size_t(0), size_t(4), size_t(6), size_t(12), size_t(20),
                       size_t(24), size_t(32), size_t(40), size_t(48),
                       size_t(56), size_t(60), size_t(64), encoded.size()-1}) {
        auto broken = encoded;
        broken.at(pos) ^= 0x40;
        rejects([&] { (void)decode_final_result(broken); }, "corrupted header/body refused");
    }
    for (size_t cut : {size_t(0), size_t(1), size_t(63), size_t(64), encoded.size()-1}) {
        auto broken = encoded;
        broken.resize(cut);
        rejects([&] { (void)decode_final_result(broken); }, "truncated result rejected");
    }
    auto broken = encoded;
    broken.push_back(0);
    rejects([&] { (void)decode_final_result(broken); }, "trailing bytes refused");
    broken = encoded;
    broken.at(6) = 2;
    rejects([&] { (void)decode_final_result(broken); }, "unsupported feature bits rejected");
    broken = encoded;
    broken.at(4) = 2;
    rejects([&] { (void)decode_final_result(broken); }, "unknown version refused");
}
void gate_tests() {
    FinalResultGate gate;
    const auto r = sample(false);
    rejects([&] { (void)gate.provisional(); }, "cannot read an empty provisional slot");
    rejects([&] { (void)gate.take_committed(); }, "cannot publish without a commit");
    gate.stage(r, 2, 120, 8, r.plan_fingerprint, 3);
    ok(gate.has_provisional(), "staged result is visible as provisional");
    rejects([&] { (void)gate.take_committed(); }, "cannot publish unacknowledged result");
    rejects([&] { gate.stage(r, 2, 120, 8, r.plan_fingerprint, 3); }, "overlapping result rejected");
    gate.mark_committed(120, 8);
    ok(!gate.has_provisional(), "successful commit marks result ready for consumption");
    rejects([&] { gate.mark_committed(120, 8); }, "double commit refused");
    raw_equal(r, gate.take_committed());
    rejects([&] { (void)gate.take_committed(); }, "result delivered only once");
    gate.stage(r, 2, 120, 8, r.plan_fingerprint, 3);
    rejects([&] { gate.mark_committed(120, 9); }, "mismatched commit poisons result gate");
    ok(gate.poisoned(), "bad commit permanently poisons slot");
    rejects([&] { gate.stage(r, 2, 120, 8, r.plan_fingerprint, 3); }, "poisoned gate cannot recover");

    FinalResultGate bad;
    rejects([&] { bad.stage(r, 1, 120, 8, r.plan_fingerprint, 3); }, "bad result identity poisons staging");
    ok(bad.poisoned(), "mismatched result fails closed");
}
void socket_tests() {
    int sockets[2];
    ok(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "create local socketpair");
    const auto r = sample();
    send_final_result(sockets[0], r);
    auto response = receive_final_result(sockets[1]);
    ok(response.has_value(), "receive final result over AF_UNIX");
    raw_equal(r, *response);
    ok(::shutdown(sockets[0], SHUT_WR) == 0, "close sender gracefully");
    ok(!receive_final_result(sockets[1]).has_value(), "clean socket EOF returns nullopt");
    ::close(sockets[0]); ::close(sockets[1]);

    ok(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "create timeout pair");
    rejects([&] { (void)receive_final_result(sockets[0], 20); }, "reader has bounded timeout");
    rejects([&] { (void)receive_final_result(sockets[0], -1); }, "negative timeout refused");
    ::close(sockets[0]); ::close(sockets[1]);

    ok(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "create truncated stream pair");
    auto encoded = encode_final_result(r);
    ok(::send(sockets[0], encoded.data(), final_result_header_bytes + 3, MSG_NOSIGNAL) ==
       static_cast<ssize_t>(final_result_header_bytes + 3), "transmit header and partial payload");
    ::shutdown(sockets[0], SHUT_WR);
    rejects([&] { (void)receive_final_result(sockets[1], 100); }, "partial payload refused");
    ::close(sockets[0]); ::close(sockets[1]);

    ok(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "create fragmented stream pair");
    const pid_t child = ::fork();
    ok(child >= 0, "fork fragmented payload sender");
    if (child == 0) {
        ::close(sockets[0]);
        auto encoded_child = encode_final_result(sample(false));
        for (uint8_t byte : encoded_child) if (::send(sockets[1], &byte, 1, MSG_NOSIGNAL) != 1) ::_exit(2);
        ::close(sockets[1]);
        ::_exit(0);
    }
    ::close(sockets[1]);
    auto fragmented = receive_final_result(sockets[0], 1000);
    ok(fragmented.has_value(), "fragmented result assembled");
    raw_equal(sample(false), *fragmented);
    int status = 0;
    ok(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
       "child exited cleanly after fragmented result");
    ::close(sockets[0]);
}
} // namespace
int main() {
    try {
        encoding_tests();
        gate_tests();
        socket_tests();
        std::cout << "final result protocol tests: PASS (" << assertions
                  << " assertions; CPU-only, no CUDA/HIP)\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "final result protocol tests: FAIL: " << ex.what() << '\n';
        return 1;
    }
}
