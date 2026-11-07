#include "strata_heterogeneous/shared.hpp"

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstring>
#include <fcntl.h>
#include <linux/memfd.h>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

namespace strata::heterogeneous {
namespace {
using Clock = std::chrono::steady_clock;
constexpr uint8_t transfer_marker = 0xA5;
constexpr int required_seals = F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;

std::string failure(const char* what) { return std::string(what) + ": " + std::strerror(errno); }
Clock::time_point end_time(int ms) {
    if (ms < 0) throw std::invalid_argument("negative shared handoff timeout");
    return Clock::now() + std::chrono::milliseconds(ms);
}
void wait_socket(int fd, short requested, Clock::time_point until) {
    while (true) {
        const auto remain = std::chrono::duration_cast<std::chrono::milliseconds>(until - Clock::now()).count();
        if (remain < 0) throw std::runtime_error("shared descriptor transfer timed out");
        pollfd pfd{fd, requested, 0};
        const int rc = ::poll(&pfd, 1, static_cast<int>(std::min<int64_t>(remain, INT_MAX)));
        if (rc > 0) {
            if (pfd.revents & POLLNVAL) throw std::runtime_error("shared descriptor invalid socket");
            return;
        }
        if (rc == 0) throw std::runtime_error("shared descriptor transfer timed out");
        if (errno != EINTR) throw std::runtime_error(failure("shared descriptor poll"));
    }
}
uint32_t checksum_bytes(const uint8_t* values, size_t n) {
    uint32_t result = 2166136261u;
    for (size_t i = 0; i < n; ++i) { result ^= values[i]; result *= 16777619u; }
    return result;
}
} // namespace

SharedActivation::SharedActivation(int fd) : fd_(fd) {
    struct stat st {};
    if (::fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size != static_cast<off_t>(capacity))
        throw std::runtime_error("shared activation has incorrect memory object or capacity");
    const int seals = ::fcntl(fd_, F_GET_SEALS);
    if (seals < 0 || (seals & required_seals) != required_seals)
        throw std::runtime_error("shared activation memory is not size-sealed");
    void* p = ::mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) throw std::runtime_error(failure("shared activation mmap"));
    bytes_ = static_cast<uint8_t*>(p);
}
SharedActivation::~SharedActivation() noexcept { release(); }
void SharedActivation::release() noexcept {
    if (bytes_) { (void) ::munmap(bytes_, capacity); bytes_ = nullptr; }
    if (fd_ >= 0) { (void) ::close(fd_); fd_ = -1; }
}
SharedActivation::SharedActivation(SharedActivation&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)), bytes_(std::exchange(other.bytes_, nullptr)) {}
SharedActivation& SharedActivation::operator=(SharedActivation&& other) noexcept {
    if (this != &other) {
        release(); fd_ = std::exchange(other.fd_, -1); bytes_ = std::exchange(other.bytes_, nullptr);
    }
    return *this;
}
SharedActivation SharedActivation::create() {
    const int fd = static_cast<int>(::syscall(SYS_memfd_create, "strata-heterogeneous-activation",
                                               MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (fd < 0) throw std::runtime_error(failure("memfd_create"));
    try {
        if (::ftruncate(fd, capacity) != 0) throw std::runtime_error(failure("shared activation ftruncate"));
        if (::fcntl(fd, F_ADD_SEALS, required_seals) != 0)
            throw std::runtime_error(failure("shared activation seals"));
        return SharedActivation(fd);
    } catch (...) { (void) ::close(fd); throw; }
}
void SharedActivation::send(int fd, int timeout_ms) const {
    if (!valid()) throw std::logic_error("attempted to transfer an invalid shared activation");
    const auto until = end_time(timeout_ms);
    char one = static_cast<char>(transfer_marker);
    iovec iov{&one, 1};
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))]{};
    msghdr message{};
    message.msg_iov = &iov; message.msg_iovlen = 1;
    message.msg_control = control; message.msg_controllen = sizeof(control);
    cmsghdr* cmsg = CMSG_FIRSTHDR(&message);
    cmsg->cmsg_level = SOL_SOCKET; cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(cmsg), &fd_, sizeof(fd_));
    while (true) {
        if (Clock::now() > until) throw std::runtime_error("shared descriptor transfer timed out");
        const ssize_t n = ::sendmsg(fd, &message, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n == 1) return;
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            wait_socket(fd, POLLOUT, until); continue;
        }
        throw std::runtime_error(n < 0 ? failure("shared descriptor sendmsg") : "short shared descriptor send");
    }
}
SharedActivation SharedActivation::receive(int fd, int timeout_ms) {
    const auto until = end_time(timeout_ms);
    char one = 0;
    iovec iov{&one, 1};
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int) * 2)]{};
    msghdr message{};
    message.msg_iov = &iov; message.msg_iovlen = 1;
    message.msg_control = control; message.msg_controllen = sizeof(control);
    while (true) {
        if (Clock::now() > until) throw std::runtime_error("shared descriptor transfer timed out");
        const ssize_t n = ::recvmsg(fd, &message, MSG_CMSG_CLOEXEC | MSG_DONTWAIT);
        if (n == 1) break;
        if (n == 0) throw std::runtime_error("worker socket closed before shared descriptor");
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            wait_socket(fd, POLLIN, until); continue;
        }
        throw std::runtime_error(n < 0 ? failure("shared descriptor recvmsg") : "bad shared descriptor marker");
    }
    int received = -1;
    size_t count = 0;
    for (cmsghdr* cmsg = CMSG_FIRSTHDR(&message); cmsg; cmsg = CMSG_NXTHDR(&message, cmsg)) {
        if (cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) continue;
        if (cmsg->cmsg_len < CMSG_LEN(0)) continue;
        const size_t bytes = cmsg->cmsg_len - CMSG_LEN(0);
        if (bytes % sizeof(int) != 0) continue;
        for (size_t at = 0; at < bytes; at += sizeof(int)) {
            int attached = -1;
            std::memcpy(&attached, CMSG_DATA(cmsg) + at, sizeof(attached));
            if (count++ == 0) received = attached;
            else (void) ::close(attached);
        }
    }
    if ((message.msg_flags & MSG_CTRUNC) || count != 1 ||
        static_cast<uint8_t>(one) != transfer_marker) {
        if (received >= 0) (void) ::close(received);
        throw std::runtime_error("missing, malformed or extra shared descriptor");
    }
    try { return SharedActivation(received); }
    catch (...) { (void) ::close(received); throw; }
}
void SharedActivation::store(const std::vector<float>& values) {
    if (!valid()) throw std::logic_error("shared activation is not mapped");
    if (values.empty() || values.size() > capacity / sizeof(float))
        throw std::invalid_argument("shared activation exceeds capacity or is empty");
    std::memcpy(bytes_, values.data(), values.size() * sizeof(float));
}
std::vector<float> SharedActivation::load(size_t count) const {
    if (!valid()) throw std::logic_error("shared activation is not mapped");
    if (count == 0 || count > capacity / sizeof(float))
        throw std::invalid_argument("shared activation load exceeds capacity or is empty");
    std::vector<float> out(count);
    std::memcpy(out.data(), bytes_, count * sizeof(float));
    return out;
}
uint32_t SharedActivation::checksum(size_t bytes) const {
    if (!valid()) throw std::logic_error("shared activation is not mapped");
    if (bytes == 0 || bytes > capacity) throw std::invalid_argument("shared checksum range invalid");
    return checksum_bytes(bytes_, bytes);
}
} // namespace strata::heterogeneous
