#include "strata/kernels/handoff_guard.hpp"
#include <cstdlib>
#include <iostream>
using namespace strata::kernels::handoff_guard;

int main() {
    static_assert(finite_f32_bits(0) && finite_f32_bits(0x80000000u));
    static_assert(finite_f32_bits(0x7f7fffffu) && finite_f32_bits(0xff7fffffu));
    static_assert(finite_f32_bits(1) && finite_f32_bits(0x807fffffu));
    static_assert(!finite_f32_bits(0x7f800000u) && !finite_f32_bits(0xff800000u));
    static_assert(!finite_f32_bits(0x7fc00000u) && !finite_f32_bits(0xff800001u));
    static_assert(!accepted(pending) && accepted(ready) && !accepted(nonfinite) && !accepted(aborted));
    // Atomic-add order is the only order the last block uses. Exhaustively put
    // one bad contribution in each possible position, plus all-good/all-bad.
    for (uint32_t blocks = 1; blocks <= max_blocks; ++blocks) {
        for (int bad_at = -2; bad_at < static_cast<int>(blocks); ++bad_at) {
            uint32_t counter = 0;
            for (uint32_t i = 0; i < blocks; ++i) {
                const bool bad = bad_at == -2 || bad_at == static_cast<int>(i);
                const auto previous = counter;
                counter += contribution(bad);
                if (last_block(previous, blocks) != (i + 1u == blocks)) return EXIT_FAILURE;
                if (i + 1u == blocks && verdict(previous, bad) != (bad_at == -1 ? ready : nonfinite))
                    return EXIT_FAILURE;
            }
            if ((counter & block_mask) != blocks) return EXIT_FAILURE;
        }
    }
    std::cout << "guarded handoff CPU policy: PASS\n";
}
