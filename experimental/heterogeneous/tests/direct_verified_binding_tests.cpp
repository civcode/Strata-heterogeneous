#include "strata_heterogeneous/direct_verified_binding.hpp"
#include <array>
#include <stdexcept>
using namespace strata::heterogeneous;

void check(bool ok) { if (!ok) throw std::runtime_error("direct binding check failed"); }
int main() {
    alignas(16) std::array<float, 16> storage{}, device{};
    uint32_t ready = 0, alias = 0;
    auto* bytes = reinterpret_cast<uint8_t*>(storage.data());
    DirectVerifiedBinding b{bytes, device.data(), &ready, &alias, sizeof(storage)};
    check(static_cast<bool>(b));
    check(b.matches(bytes, 16) && b.matches(bytes, sizeof(storage)));
    check(!b.matches(bytes + 4, 16) && !b.matches(bytes, 0));
    check(!b.matches(bytes, 3) && !b.matches(bytes, sizeof(storage) + 4));
    check(!b.published());
    ready = 1; check(b.published());
    ready = 2; check(!b.published());
    ready = 0; check(!b.published());
    b.host_ready = nullptr; check(!b.published() && !b.matches(bytes, 16));
    check(!DirectVerifiedBinding{}.matches(bytes, 16));
}
