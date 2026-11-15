#include "strata_heterogeneous/long_context.hpp"
#include <cstdint>
#include <iostream>
using namespace strata::heterogeneous;
int main() {
    static_assert(long_context_cells_valid(2048) && long_context_cells_valid(262144));
    static_assert(!long_context_cells_valid(262145) && !long_context_cells_valid(0));
    static_assert(!long_context_cells_valid(65535));
    static_assert(long_context_stage_valid(262144, 1ull<<30, 0, 8, 24*512, 24));
    static_assert(!long_context_stage_valid(262144, (1ull<<30)-1, 0, 8, 24*512, 24));
    static_assert(!long_context_stage_valid(262144, 1ull<<30, 1, 8, 24*512, 24));
    static_assert(!long_context_stage_valid(262144, 1ull<<30, 0, 8, 24*512-1, 24));
    static_assert(capacity_fits(100, 40, 60) && !capacity_fits(100, 40, 61));
    static_assert(!capacity_fits(0, 1, UINT64_MAX) && !capacity_fits(UINT64_MAX, 1, UINT64_MAX));
    static_assert(resident_command_cap(32768, true, false) == 384*1024);
    static_assert(resident_command_cap(8192, false, false) == 65536);
    static_assert(resident_command_cap(262144, true, true) >= 11ull * max_long_context);
    std::cout << "long-context policy checks passed\n";
}
