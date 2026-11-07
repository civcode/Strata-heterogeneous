// CPU-only Linux descriptor-passing and shared activation tests.
#include "strata_heterogeneous/shared.hpp"

#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace strata::heterogeneous;
namespace {
int assertions = 0;
void check(bool v, const char* reason) {
    ++assertions;
    if (!v) throw std::runtime_error(reason);
}
template<class F> void fails(F&& fn, const char* reason) {
    bool threw = false;
    try { fn(); } catch (const std::exception&) { threw = true; }
    check(threw, reason);
}
void send_invalid_fd(int sock, int descriptor) {
    char marker = static_cast<char>(0xA5);
    iovec iov{&marker, 1};
    alignas(cmsghdr) char buf[CMSG_SPACE(sizeof(int))]{};
    msghdr message{};
    message.msg_iov = &iov; message.msg_iovlen = 1;
    message.msg_control = buf; message.msg_controllen = sizeof(buf);
    cmsghdr* msg = CMSG_FIRSTHDR(&message);
    msg->cmsg_level = SOL_SOCKET; msg->cmsg_type = SCM_RIGHTS;
    msg->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(msg), &descriptor, sizeof(descriptor));
    if (::sendmsg(sock, &message, MSG_NOSIGNAL) != 1) throw std::runtime_error("test sendmsg failed");
}
} // namespace
int main() {
    try {
        auto shared = SharedActivation::create();
        check(shared.valid(), "memfd mapped");
        std::vector<float> data{1.f, -0.f, 0.25f, 3.f};
        const uint32_t arbitrary_nan = 0x7fc12345u;
        std::memcpy(&data[2], &arbitrary_nan, sizeof(arbitrary_nan));
        shared.store(data);
        auto roundtrip = shared.load(data.size());
        check(std::memcmp(data.data(), roundtrip.data(), data.size() * sizeof(float)) == 0,
              "shared slot retains exact float bits including NaN and signed zero");
        const uint32_t before = shared.checksum(data.size() * sizeof(float));
        check(before != 0, "nonempty checksum");
        int sockets[2]{};
        check(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socketpair");
        shared.send(sockets[0]);
        auto other = SharedActivation::receive(sockets[1]);
        check(other.valid(), "SCM_RIGHTS receives a valid memfd");
        check(other.checksum(data.size() * sizeof(float)) == before, "mappings share input bytes");
        auto other_data = other.load(data.size());
        other_data[0] = -13.25f;
        other.store(other_data);
        check(shared.load(data.size())[0] == -13.25f, "separate maps share updates");
        check(shared.checksum(data.size() * sizeof(float)) != before, "digest changes on buffer mutation");
        ::close(sockets[0]); ::close(sockets[1]);

        SharedActivation moved = std::move(other);
        check(moved.valid() && !other.valid(), "memfd move transfers ownership");
        moved = std::move(shared);
        check(moved.valid() && !shared.valid(), "memfd move assignment releases prior mapping");
        fails([&] { (void) moved.checksum(SharedActivation::capacity + 1); }, "digest bounds");
        fails([&] { (void) moved.load(SharedActivation::capacity / sizeof(float) + 1); }, "read bounds");
        fails([&] { moved.store({}); }, "reject empty slot");
        fails([&] { moved.store(std::vector<float>(SharedActivation::capacity / sizeof(float) + 1)); },
              "reject oversized slot");

        check(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "missing descriptor pair");
        const char marker = static_cast<char>(0xA5);
        check(::send(sockets[0], &marker, 1, MSG_NOSIGNAL) == 1, "send marker without descriptor");
        fails([&] { (void) SharedActivation::receive(sockets[1], 100); }, "missing descriptor rejected");
        ::close(sockets[0]); ::close(sockets[1]);

        check(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "bad descriptor pair");
        int devnull = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
        check(devnull >= 0, "open invalid mapping source");
        send_invalid_fd(sockets[0], devnull);
        fails([&] { (void) SharedActivation::receive(sockets[1], 100); }, "nonmemfd rejected");
        ::close(devnull); ::close(sockets[0]); ::close(sockets[1]);

        check(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "timeout pair");
        fails([&] { (void) SharedActivation::receive(sockets[1], 10); }, "bounded descriptor timeout");
        ::close(sockets[0]); ::close(sockets[1]);

        // Actual mapping shared across processes; subprocess test uses exec elsewhere.
        check(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "process pair");
        const pid_t pid = ::fork();
        check(pid >= 0, "fork child for CPU-only mmap test");
        if (pid == 0) {
            ::close(sockets[0]);
            try {
                auto in = SharedActivation::receive(sockets[1], 1000);
                auto values = in.load(4);
                if (values[0] != -13.25f) ::_exit(2);
                values[3] = 42.f;
                in.store(values);
                ::close(sockets[1]);
                ::_exit(0);
            } catch (...) { ::_exit(3); }
        }
        ::close(sockets[1]);
        moved.send(sockets[0], 1000);
        int status = 0;
        check(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "child maps parent memfd and edits it");
        check(moved.load(4)[3] == 42.f, "parent sees child write");
        ::close(sockets[0]);

        // Checksum disagreement in the worker must return a well-formed error,
        // not publish data or stall because an error frame retained the digest.
        const Plan plan = parse_plan("hip:0:0-4", 4, {2, 4}, 999);
        auto staging = SharedActivation::create();
        staging.store(std::vector<float>(14, 1.f));
        check(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "failure injection sockets");
        const pid_t faulty = ::fork();
        check(faulty >= 0, "fork for corrupted shared handoff");
        if (faulty == 0) {
            ::close(sockets[0]);
            try {
                MockStageExecutor executor;
                serve_worker_stateful(sockets[1], plan, 0, executor, 1000, true);
                ::close(sockets[1]);
                ::_exit(0);
            } catch (...) { ::_exit(5); }
        }
        ::close(sockets[1]);
        auto hello = receive_frame(sockets[0], 1000);
        check(hello && hello->kind == Kind::hello_shared, "worker announces shared mode");
        Frame ready = *hello;
        ready.kind = Kind::ready;
        ready.identity = {};
        send_frame(sockets[0], ready, 1000);
        staging.send(sockets[0], 1000);
        Frame corrupted;
        corrupted.kind = Kind::shared_data;
        corrupted.plan_fingerprint = plan.fingerprint();
        corrupted.rows = 1; corrupted.request_id = 7; corrupted.window_id = 1;
        corrupted.geometry = plan.geometry;
        corrupted.shared_checksum = staging.checksum(14 * sizeof(float)) ^ 0x1000u;
        send_frame(sockets[0], corrupted, 1000);
        const auto rejected = receive_frame(sockets[0], 1000);
        check(rejected && rejected->kind == Kind::failure, "corrupted shared input produces failure frame");
        check(rejected->error.find("checksum") != std::string::npos,
              "failure distinguishes mismatched shared checksum");
        check(rejected->shared_checksum == 0, "error frame clears previous shared checksum");
        check(::waitpid(faulty, &status, 0) == faulty && WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "worker cleanly stops after handoff error");
        ::close(sockets[0]);
        std::cout << "shared activation tests: PASS (" << assertions
                  << " assertions, sealed memfd + SCM_RIGHTS, no GPU)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "shared activation tests: FAIL: " << e.what() << '\n';
        return 1;
    }
}
